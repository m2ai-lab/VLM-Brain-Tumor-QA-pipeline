#!/usr/bin/env python3
"""
run_local.py — Local (non-SLURM) entrypoint for the experiment orchestrator.

Runs the same jobs as run_experiments.py but executes each testing script
directly as a subprocess using the current Python interpreter, instead of
generating and submitting sbatch files.

Intended for:
  - Local development / smoke testing
  - Any machine without a SLURM scheduler (laptop, VM, cloud instance)
  - Running the OpenAI/API-based models that don't need GPU allocation

The same filter flags as run_experiments.py are supported, so you can use
identical invocations to target the same subset of jobs.

Usage
-----
  # Run all enabled experiments sequentially
  python -m experiment_orchestrator.run_local

  # Preview jobs without running
  python -m experiment_orchestrator.run_local --list

  # Run a single model
  python -m experiment_orchestrator.run_local --model MedGemma1.5

  # Run only blank tests for all models
  python -m experiment_orchestrator.run_local --variant blank

  # Run only human tests, skipping LLaVA-Med
  python -m experiment_orchestrator.run_local --test human --exclude-model LLaVA-Med

  # Run up to 2 jobs in parallel
  python -m experiment_orchestrator.run_local --jobs 2

  # Use a custom experiment config
  python -m experiment_orchestrator.run_local --config my_experiment.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiment_orchestrator.config_schema import ExperimentSuite
from experiment_orchestrator.config_resolver import resolve_all, expand_suite_raw
from experiment_orchestrator.adapters import get_adapter
# Reuse the filter logic from run_experiments so behaviour is identical
from experiment_orchestrator.run_experiments import apply_filters, load_config

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "experiment.json")
_PRINT_LOCK = threading.Lock()  # keep output from interleaved jobs readable


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local experiment runner — executes VLM testing scripts directly\n"
            "without SLURM.  Accepts the same filter flags as run_experiments.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Filter flags (same as run_experiments.py):\n"
            "  --model MODEL [...]            Only run these model(s)\n"
            "  --test  TEST  [...]            Only run these test name(s)\n"
            "  --variant VAR [...]            Only run these variant type(s)\n"
            "  --exclude-model MODEL [...]    Skip these model(s)\n"
            "  --exclude-test  TEST  [...]    Skip these test name(s)\n"
            "  --exclude-variant VAR [...]    Skip these variant type(s)\n\n"
            "Examples:\n"
            "  %(prog)s --model MedGemma1.5\n"
            "  %(prog)s --test human --exclude-model LLaVA-Med\n"
            "  %(prog)s --variant blank --jobs 2\n"
            "  %(prog)s --list\n"
        ),
    )

    # ── Config ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to experiment.json.",
    )

    # ── Allow-list filters (identical to run_experiments.py) ─────────────────
    allow = parser.add_argument_group("allow-list filters")
    allow.add_argument("--model",   nargs="+", default=[], metavar="MODEL")
    allow.add_argument("--test",    nargs="+", default=[], metavar="TEST")
    allow.add_argument("--variant", nargs="+", default=[], metavar="VARIANT")

    # ── Deny-list filters ─────────────────────────────────────────────────────
    deny = parser.add_argument_group("deny-list filters (always applied last)")
    deny.add_argument("--exclude-model",   nargs="+", default=[], metavar="MODEL")
    deny.add_argument("--exclude-test",    nargs="+", default=[], metavar="TEST")
    deny.add_argument("--exclude-variant", nargs="+", default=[], metavar="VARIANT")

    # ── Run modes ─────────────────────────────────────────────────────────────
    modes = parser.add_argument_group("run modes")
    modes.add_argument(
        "--list", action="store_true",
        help="Print matching jobs and exit without running anything.",
    )
    modes.add_argument(
        "--dry-run", action="store_true",
        help="Print the command that would be run for each job, but don't execute it.",
    )
    modes.add_argument(
        "--jobs", type=int, default=1, metavar="N",
        help=(
            "Number of jobs to run in parallel (default: 1 = sequential). "
            "Use with caution on machines without enough RAM/GPU memory."
        ),
    )
    modes.add_argument(
        "--python", type=str, default=sys.executable,
        help=(
            "Python interpreter to use. Defaults to the currently active interpreter "
            "(i.e., whatever conda/venv you are running this script from)."
        ),
    )
    modes.add_argument(
        "--debug", action="store_true",
        help="Enable verbose debug logging.",
    )

    return parser.parse_args(argv)


# ── Job execution ─────────────────────────────────────────────────────────────

def _stream_output(proc: subprocess.Popen, prefix: str) -> None:
    """Read stdout/stderr from a process and print with a job prefix."""
    for line in proc.stdout:
        with _PRINT_LOCK:
            print(f"{prefix} {line}", end="", flush=True)


def run_job(job_name: str, command: str, python_exe: str, dry_run: bool) -> dict:
    """
    Execute a single testing script as a subprocess.

    The adapter's build_command() returns   /path/to/script.py --arg val ...
    We prepend the Python interpreter so it becomes:
        python /path/to/script.py --arg val ...

    On Windows, posixpath separators in the script path are normalised before
    passing to the shell.

    Returns a dict with keys: job_name, success, elapsed, returncode.
    """
    # Normalise path separators for the current OS
    parts  = shlex.split(command)
    script = str(Path(parts[0]))          # convert / → \ on Windows
    args   = parts[1:]
    cmd    = [python_exe, script] + args

    prefix = f"[{job_name}]"
    cmd_str = " ".join(cmd)

    if dry_run:
        print(f"{prefix} DRY-RUN: {cmd_str}", flush=True)
        return {"job_name": job_name, "success": True, "elapsed": 0.0, "returncode": 0}

    with _PRINT_LOCK:
        print(f"\n{'─' * 60}", flush=True)
        print(f"{prefix} Starting", flush=True)
        print(f"{prefix} Command: {cmd_str}", flush=True)
        print(f"{'─' * 60}", flush=True)

    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,                    # line-buffered
        )
        _stream_output(proc, prefix)
        proc.wait()
        elapsed = time.monotonic() - t0
        success = proc.returncode == 0

        with _PRINT_LOCK:
            status = "✓ DONE" if success else f"✗ FAILED (exit {proc.returncode})"
            print(f"{prefix} {status} in {elapsed:.1f}s", flush=True)

        return {
            "job_name": job_name,
            "success": success,
            "elapsed": elapsed,
            "returncode": proc.returncode,
        }

    except Exception as exc:
        elapsed = time.monotonic() - t0
        with _PRINT_LOCK:
            print(f"{prefix} ✗ ERROR: {exc}", flush=True)
        return {
            "job_name": job_name,
            "success": False,
            "elapsed": elapsed,
            "returncode": -1,
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── 1. Load config ────────────────────────────────────────────────────
    try:
        suite = load_config(args.config)
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        return 1

    # ── 2. Resolve jobs ───────────────────────────────────────────────────
    all_jobs = resolve_all(suite)
    logger.info("Resolved %d total jobs from config", len(all_jobs))

    # ── 3. Apply filters (reuses run_experiments.apply_filters) ──────────
    jobs = apply_filters(all_jobs, args)

    if not jobs:
        logger.warning("No jobs to run after filtering.")
        return 0

    # ── 4. --list mode ────────────────────────────────────────────────────
    if args.list:
        col_w = max(len(j.job_name) for j in jobs) + 2
        print(f"\n{'Job':>{col_w}}   {'Model':<30} {'Test':<30} {'Variant'}")
        print("-" * (col_w + 75))
        for j in jobs:
            print(f"{j.job_name:>{col_w}}   {j.model_name:<30} {j.test_name:<30} {j.variant}")
        print(f"\n{len(jobs)} job(s) matched.")
        return 0

    # ── 5. Build commands ─────────────────────────────────────────────────
    job_commands: list[tuple[str, str]] = []   # (job_name, command_string)

    for job in jobs:
        adapter = get_adapter(job.adapter_name)
        try:
            adapter.validate(job)
        except ValueError as e:
            logger.error("Validation failed for %s: %s", job.job_name, e)
            return 1

        command = adapter.build_command(job, suite.global_config.project_root)
        job_commands.append((job.job_name, command))

    # ── 6. Execute ────────────────────────────────────────────────────────
    n_jobs  = len(job_commands)
    python  = args.python
    dry_run = args.dry_run
    n_workers = max(1, args.jobs)

    print(f"\nRunning {n_jobs} job(s) with {n_workers} worker(s) "
          f"using {python}", flush=True)
    if dry_run:
        print("(DRY-RUN mode — commands will be printed but not executed)", flush=True)

    results = []

    if n_workers == 1:
        # Sequential — simple loop, easiest to read output
        for job_name, command in job_commands:
            result = run_job(job_name, command, python, dry_run)
            results.append(result)
    else:
        # Parallel — use a thread pool (each job runs a subprocess)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(run_job, jn, cmd, python, dry_run): jn
                for jn, cmd in job_commands
            }
            for future in as_completed(futures):
                results.append(future.result())

    # ── 7. Summary ────────────────────────────────────────────────────────
    passed  = [r for r in results if r["success"]]
    failed  = [r for r in results if not r["success"]]
    total_t = sum(r["elapsed"] for r in results)

    print(f"\n{'═' * 60}", flush=True)
    print(f"  Results: {len(passed)}/{n_jobs} jobs succeeded  "
          f"(wall time: {total_t:.1f}s)", flush=True)

    if failed:
        print(f"\n  Failed jobs:", flush=True)
        for r in failed:
            print(f"    ✗  {r['job_name']}  (exit {r['returncode']})", flush=True)
        print(f"{'═' * 60}\n", flush=True)
        return 1

    print(f"{'═' * 60}\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
