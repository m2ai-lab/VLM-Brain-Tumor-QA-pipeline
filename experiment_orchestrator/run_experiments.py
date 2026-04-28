#!/usr/bin/env python3
"""
run_experiments.py — Main entrypoint for the experiment orchestrator.

Reads experiment.json, resolves all jobs, generates sbatch scripts,
and submits them via the SLURM scheduler with concurrency management.

Usage:
    python -m experiment_orchestrator.run_experiments
    python -m experiment_orchestrator.run_experiments --only MedGemma1.5 Qwen2.5
    python -m experiment_orchestrator.run_experiments --exclude LLaVA-Med
    python -m experiment_orchestrator.run_experiments --exclude-name human single_slice_shuffled
    python -m experiment_orchestrator.run_experiments --dry-run
    python -m experiment_orchestrator.run_experiments --config custom.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Ensure the project root is on sys.path so this script works when invoked
# directly (e.g., `python run_experiments.py`) rather than as a module.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiment_orchestrator.config_schema import ExperimentSuite
from experiment_orchestrator.config_resolver import resolve_all, expand_suite_raw
from experiment_orchestrator.adapters import get_adapter
from experiment_orchestrator.slurm_template import write_sbatch_file
from experiment_orchestrator.slurm_scheduler import SlurmScheduler

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiment.json",
)
DEFAULT_GENERATED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "generated_slurm",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment Orchestrator — run VLM experiments via SLURM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to experiment.json config file.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=[],
        help="Only run experiments for these model names (e.g., MedGemma1.5 Qwen2.5).",
    )
    parser.add_argument(
        "--name",
        nargs="+",
        default=[],
        help="Only run experiments for specific test names (e.g., human, single_slice_shuffled).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Only run experiments for a specific variant type (e.g., blank, single_slice).",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=[],
        help="Run specific model:test combinations (e.g., MedGemma1.5:single_slice_shuffled).",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="Skip experiments for these model names (e.g., LLaVA-Med MedImageInsight). "
             "Applied after all --only/--name/--variant/--include filters.",
    )
    parser.add_argument(
        "--exclude-name",
        nargs="+",
        default=[],
        help="Skip experiments for these test names (e.g., human single_slice_shuffled). "
             "Applied after all --only/--name/--variant/--include filters.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate sbatch files but do not submit to SLURM.",
    )
    parser.add_argument(
        "--generated-dir",
        type=str,
        default=DEFAULT_GENERATED_DIR,
        help="Directory to write generated .sbatch files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def load_config(config_path: str) -> ExperimentSuite:
    """Load, expand {variable} placeholders, and validate experiment.json through Pydantic."""
    logger.info("Loading config from %s", config_path)

    with open(config_path, "r") as f:
        raw = json.load(f)

    # Expand {variable} placeholders from config.yaml before Pydantic validation
    raw = expand_suite_raw(raw)

    suite = ExperimentSuite.model_validate(raw)
    logger.info(
        "Config loaded: %d models, %d environments",
        len(suite.models),
        len(suite.environments),
    )
    return suite


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
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

    # Keep a copy of all jobs for the additive --include filter
    original_all_jobs = list(all_jobs)

    # Subtractive (AND) filters
    if args.only:
        only_set = set(args.only)
        all_jobs = [j for j in all_jobs if j.model_name in only_set]
        logger.info("Filtered to %d jobs matching --only %s", len(all_jobs), args.only)
        
    if args.name:
        name_set = set(args.name)
        all_jobs = [j for j in all_jobs if j.test_name in name_set]
        logger.info("Filtered to %d jobs matching --name %s", len(all_jobs), args.name)

    if args.variant:
        all_jobs = [j for j in all_jobs if j.variant == args.variant]
        logger.info("Filtered to %d jobs matching --variant %s", len(all_jobs), args.variant)
        
    # Additive (OR) filter
    if args.include:
        included_set = set(args.include)
        included_jobs = {
            j.job_name: j for j in original_all_jobs
            if f"{j.model_name}:{j.test_name}" in included_set
        }
        
        # Combine the subtractive filtered jobs AND the explicitly included jobs
        final_jobs = {j.job_name: j for j in all_jobs}
        final_jobs.update(included_jobs)
        all_jobs = list(final_jobs.values())
        logger.info("Final job list contains %d jobs after applying --include", len(all_jobs))

    # Exclusion (denylist) filters — applied last so they always win
    if args.exclude:
        exclude_set = set(args.exclude)
        before = len(all_jobs)
        all_jobs = [j for j in all_jobs if j.model_name not in exclude_set]
        logger.info(
            "--exclude removed %d job(s) matching %s; %d remaining",
            before - len(all_jobs), args.exclude, len(all_jobs),
        )

    if args.exclude_name:
        exclude_name_set = set(args.exclude_name)
        before = len(all_jobs)
        all_jobs = [j for j in all_jobs if j.test_name not in exclude_name_set]
        logger.info(
            "--exclude-name removed %d job(s) matching %s; %d remaining",
            before - len(all_jobs), args.exclude_name, len(all_jobs),
        )

    if not all_jobs:
        logger.warning("No jobs to run. Check that models are enabled in the config.")
        return 0

    # ── 3. Validate & generate sbatch files ───────────────────────────────
    sbatch_jobs: list[tuple[str, str]] = []  # (job_name, sbatch_path)

    for job in all_jobs:
        # Validate via adapter
        adapter = get_adapter(job.adapter_name)
        try:
            adapter.validate(job)
        except ValueError as e:
            logger.error("Validation failed for %s: %s", job.job_name, e)
            return 1

        # Build the command
        command = adapter.build_command(job, suite.global_config.project_root)

        # Look up environment config
        if job.environment not in suite.environments:
            logger.error(
                "Environment '%s' not found for job '%s'. Available: %s",
                job.environment, job.job_name, list(suite.environments.keys()),
            )
            return 1
        env_cfg = suite.environments[job.environment]

        # Generate sbatch file
        sbatch_path = write_sbatch_file(
            job=job,
            env_cfg=env_cfg,
            command=command,
            output_dir=args.generated_dir,
        )
        sbatch_jobs.append((job.job_name, sbatch_path))
        logger.info("Generated: %s", sbatch_path)

    logger.info(
        "Generated %d sbatch files in %s", len(sbatch_jobs), args.generated_dir
    )

    # ── 4. Submit via scheduler ───────────────────────────────────────────
    scheduler = SlurmScheduler(
        max_concurrent=suite.global_config.max_concurrent_jobs,
        poll_interval=suite.global_config.poll_interval_seconds,
        dry_run=args.dry_run,
    )

    result = scheduler.submit_all(sbatch_jobs)

    # ── 5. Summary ────────────────────────────────────────────────────────
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
