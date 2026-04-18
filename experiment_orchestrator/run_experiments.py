#!/usr/bin/env python3
"""
run_experiments.py — Main entrypoint for the experiment orchestrator.

Reads experiment.json, resolves all jobs, generates sbatch scripts,
and submits them via the SLURM scheduler with concurrency management.

Usage:
    python -m experiment_orchestrator.run_experiments
    python -m experiment_orchestrator.run_experiments --only MedGemma1.5 Qwen2.5
    python -m experiment_orchestrator.run_experiments --dry-run
    python -m experiment_orchestrator.run_experiments --config custom.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from experiment_orchestrator.config_schema import ExperimentSuite
from experiment_orchestrator.config_resolver import resolve_all
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
    """Load and validate experiment.json through Pydantic."""
    logger.info("Loading config from %s", config_path)

    with open(config_path, "r") as f:
        raw = json.load(f)

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

    # Filter by --only if specified
    if args.only:
        only_set = set(args.only)
        all_jobs = [j for j in all_jobs if j.model_name in only_set]
        logger.info("Filtered to %d jobs matching --only %s", len(all_jobs), args.only)

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
