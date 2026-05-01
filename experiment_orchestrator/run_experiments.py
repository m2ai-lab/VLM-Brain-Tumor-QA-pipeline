#!/usr/bin/env python3
"""
run_experiments.py — Main entrypoint for the experiment orchestrator.

Reads experiment.json, resolves all jobs, generates sbatch scripts,
and submits them via the SLURM scheduler with concurrency management.

Filtering
---------
All filter flags are optional and compose with AND logic.
Allow-list flags restrict to matching jobs; deny-list flags remove them.
Deny-list flags always win (applied last).

Allow-list flags
  --model    MODEL [MODEL ...]     Only run these model(s)
  --test     TEST  [TEST  ...]     Only run tests with these name(s)
  --variant  VAR   [VAR   ...]     Only run jobs with these variant type(s)

Deny-list flags
  --exclude-model    MODEL [...]   Skip these model(s)
  --exclude-test     TEST  [...]   Skip tests with these name(s)
  --exclude-variant  VAR   [...]   Skip jobs with these variant type(s)

Examples
--------
  # Run every enabled experiment
  python -m experiment_orchestrator.run_experiments

  # Run a single model
  python -m experiment_orchestrator.run_experiments --model MedGemma1.5

  # Run two models
  python -m experiment_orchestrator.run_experiments --model MedGemma1.5 Qwen2.5

  # Run only the human-dataset tests across all models
  python -m experiment_orchestrator.run_experiments --test human

  # Run only blank (control) tests
  python -m experiment_orchestrator.run_experiments --variant blank

  # Run all MedGemma tests except the human benchmark
  python -m experiment_orchestrator.run_experiments --model MedGemma1.5 --exclude-test human

  # Run everything except LLaVA-Med
  python -m experiment_orchestrator.run_experiments --exclude-model LLaVA-Med

  # Preview all jobs without submitting
  python -m experiment_orchestrator.run_experiments --list

  # Dry-run (generate sbatch files but don't submit)
  python -m experiment_orchestrator.run_experiments --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiment_orchestrator.config_schema import ExperimentSuite
from experiment_orchestrator.config_resolver import resolve_all, expand_suite_raw
from experiment_orchestrator.adapters import get_adapter
from experiment_orchestrator.slurm_template import write_sbatch_file
from experiment_orchestrator.slurm_scheduler import SlurmScheduler

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "experiment.json")
DEFAULT_GENERATED_DIR = os.path.join(_PROJECT_ROOT, "generated_slurm")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment Orchestrator — submit VLM experiments to SLURM.\n"
            "All filter flags are optional and compose with AND logic.\n"
            "Deny-list flags (--exclude-*) always win over allow-list flags."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Run all enabled experiments:\n"
            "    %(prog)s\n\n"
            "  Run a single model:\n"
            "    %(prog)s --model MedGemma1.5\n\n"
            "  Run only human tests across all models:\n"
            "    %(prog)s --test human\n\n"
            "  Run only blank (control) variant:\n"
            "    %(prog)s --variant blank\n\n"
            "  Run MedGemma but skip the human test:\n"
            "    %(prog)s --model MedGemma1.5 --exclude-test human\n\n"
            "  Run everything except LLaVA-Med:\n"
            "    %(prog)s --exclude-model LLaVA-Med\n\n"
            "  Preview all matching jobs without submitting:\n"
            "    %(prog)s --list\n"
        ),
    )

    # ── Config / output ───────────────────────────────────────────────────────
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to experiment.json config file.",
    )
    parser.add_argument(
        "--generated-dir", type=str, default=DEFAULT_GENERATED_DIR,
        help="Directory to write generated .sbatch files.",
    )

    # ── Allow-list filters ────────────────────────────────────────────────────
    allow = parser.add_argument_group("allow-list filters (restrict to matching jobs)")
    allow.add_argument(
        "--model", nargs="+", default=[], metavar="MODEL",
        help="Only run jobs for these model name(s).  e.g. --model MedGemma1.5 Qwen2.5",
    )
    allow.add_argument(
        "--test", nargs="+", default=[], metavar="TEST",
        help="Only run jobs whose test name matches.  e.g. --test human single_slice",
    )
    allow.add_argument(
        "--variant", nargs="+", default=[], metavar="VARIANT",
        help="Only run jobs with these variant type(s).  e.g. --variant blank single_slice",
    )

    # ── Deny-list filters ─────────────────────────────────────────────────────
    deny = parser.add_argument_group("deny-list filters (always applied last, always win)")
    deny.add_argument(
        "--exclude-model", nargs="+", default=[], metavar="MODEL",
        help="Skip jobs for these model name(s).  e.g. --exclude-model LLaVA-Med",
    )
    deny.add_argument(
        "--exclude-test", nargs="+", default=[], metavar="TEST",
        help="Skip jobs whose test name matches.  e.g. --exclude-test human",
    )
    deny.add_argument(
        "--exclude-variant", nargs="+", default=[], metavar="VARIANT",
        help="Skip jobs with these variant type(s).  e.g. --exclude-variant blank",
    )

    # ── Run modes ─────────────────────────────────────────────────────────────
    modes = parser.add_argument_group("run modes")
    modes.add_argument(
        "--list", action="store_true",
        help="Print all matching jobs and exit without submitting or generating files.",
    )
    modes.add_argument(
        "--dry-run", action="store_true",
        help="Generate sbatch files but do not submit to SLURM.",
    )
    modes.add_argument(
        "--debug", action="store_true",
        help="Enable verbose debug logging.",
    )

    return parser.parse_args(argv)


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> ExperimentSuite:
    """Load, expand {variable} placeholders, and validate experiment.json."""
    logger.info("Loading config from %s", config_path)
    with open(config_path, "r") as f:
        raw = json.load(f)
    raw = expand_suite_raw(raw)
    suite = ExperimentSuite.model_validate(raw)
    logger.info(
        "Config loaded: %d models, %d environments",
        len(suite.models), len(suite.environments),
    )
    return suite


# ── Filtering ─────────────────────────────────────────────────────────────────

def apply_filters(jobs, args) -> list:
    """
    Apply allow-list then deny-list filters in a clear, predictable order.

    Allow-list (AND): if a flag is set, keep only jobs that match ALL given flags.
    Deny-list  (AND): remove jobs that match ANY deny flag. Always wins.
    """
    # ── Allow-list ────────────────────────────────────────────────────────────
    if args.model:
        model_set = set(args.model)
        jobs = [j for j in jobs if j.model_name in model_set]
        logger.info("--model: %d jobs match %s", len(jobs), args.model)

    if args.test:
        test_set = set(args.test)
        jobs = [j for j in jobs if j.test_name in test_set]
        logger.info("--test: %d jobs match %s", len(jobs), args.test)

    if args.variant:
        variant_set = set(args.variant)
        jobs = [j for j in jobs if j.variant in variant_set]
        logger.info("--variant: %d jobs match %s", len(jobs), args.variant)

    # ── Deny-list (always applied last) ───────────────────────────────────────
    if args.exclude_model:
        before = len(jobs)
        ex_set = set(args.exclude_model)
        jobs = [j for j in jobs if j.model_name not in ex_set]
        logger.info("--exclude-model: removed %d jobs", before - len(jobs))

    if args.exclude_test:
        before = len(jobs)
        ex_set = set(args.exclude_test)
        jobs = [j for j in jobs if j.test_name not in ex_set]
        logger.info("--exclude-test: removed %d jobs", before - len(jobs))

    if args.exclude_variant:
        before = len(jobs)
        ex_set = set(args.exclude_variant)
        jobs = [j for j in jobs if j.variant not in ex_set]
        logger.info("--exclude-variant: removed %d jobs", before - len(jobs))

    return jobs


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── 1. Load & validate config ─────────────────────────────────────────
    try:
        suite = load_config(args.config)
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        return 1

    # ── 2. Resolve hierarchy into flat job list ───────────────────────────
    all_jobs = resolve_all(suite)
    logger.info("Resolved %d total jobs from config", len(all_jobs))

    # ── 3. Apply filters ──────────────────────────────────────────────────
    jobs = apply_filters(all_jobs, args)

    if not jobs:
        logger.warning("No jobs to run after filtering. Check your filter flags and that models are enabled.")
        return 0

    # ── 4. --list mode: just print and exit ───────────────────────────────
    if args.list:
        col_w = max(len(j.job_name) for j in jobs) + 2
        print(f"\n{'Job':>{col_w}}   {'Model':<30} {'Test':<30} {'Variant'}")
        print("-" * (col_w + 75))
        for j in jobs:
            print(f"{j.job_name:>{col_w}}   {j.model_name:<30} {j.test_name:<30} {j.variant}")
        print(f"\n{len(jobs)} job(s) matched.")
        return 0

    # ── 5. Validate & generate sbatch files ───────────────────────────────
    sbatch_jobs: list[tuple[str, str]] = []

    for job in jobs:
        adapter = get_adapter(job.adapter_name)
        try:
            adapter.validate(job)
        except ValueError as e:
            logger.error("Validation failed for %s: %s", job.job_name, e)
            return 1

        command = adapter.build_command(job, suite.global_config.project_root)

        if job.environment not in suite.environments:
            logger.error(
                "Environment '%s' not found for job '%s'. Available: %s",
                job.environment, job.job_name, list(suite.environments.keys()),
            )
            return 1

        env_cfg = suite.environments[job.environment]
        sbatch_path = write_sbatch_file(
            job=job,
            env_cfg=env_cfg,
            command=command,
            output_dir=args.generated_dir,
        )
        sbatch_jobs.append((job.job_name, sbatch_path))
        logger.info("Generated: %s", sbatch_path)

    logger.info("Generated %d sbatch files in %s", len(sbatch_jobs), args.generated_dir)

    # ── 6. Submit via scheduler ───────────────────────────────────────────
    scheduler = SlurmScheduler(
        max_concurrent=suite.global_config.max_concurrent_jobs,
        poll_interval=suite.global_config.poll_interval_seconds,
        dry_run=args.dry_run,
    )

    result = scheduler.submit_all(sbatch_jobs)

    # ── 7. Summary ────────────────────────────────────────────────────────
    n_completed = len(result["completed"])
    n_failed = len(result["failed"])

    if n_failed > 0:
        logger.error(
            "%d/%d jobs failed. Check SLURM logs for details.",
            n_failed, n_completed + n_failed,
        )
        return 1

    logger.info("All %d jobs completed successfully.", n_completed)
    return 0


if __name__ == "__main__":
    sys.exit(main())

