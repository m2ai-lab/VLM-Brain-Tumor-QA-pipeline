"""
slurm_scheduler.py — Intelligent SLURM job scheduler with concurrency limits.

Submits jobs up to max_concurrent, then polls squeue at a configurable
interval. As jobs complete, new ones are submitted to fill the slots.
Tracks completed/failed jobs and prints a summary at the end.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    """Tracks a single submitted SLURM job."""
    job_id: str
    experiment_name: str
    sbatch_path: str
    status: str = "PENDING"   # PENDING → RUNNING → COMPLETED / FAILED


class SlurmScheduler:
    """
    Manages SLURM job submission with a configurable concurrency ceiling.

    Usage:
        scheduler = SlurmScheduler(max_concurrent=4, poll_interval=30)
        scheduler.submit_all([("exp_name", "/path/to/script.sbatch"), ...])
    """

    def __init__(
        self,
        max_concurrent: int = 4,
        poll_interval: int = 30,
        dry_run: bool = False,
    ):
        self.max_concurrent = max_concurrent
        self.poll_interval = poll_interval
        self.dry_run = dry_run

        self.active_jobs: dict[str, JobRecord] = {}       # job_id → record
        self.pending: deque[tuple[str, str]] = deque()     # (name, sbatch_path)
        self.completed: list[JobRecord] = []
        self.failed: list[JobRecord] = []

    # ── Public API ────────────────────────────────────────────────────────

    def submit_all(self, jobs: list[tuple[str, str]]) -> dict:
        """
        Submit all jobs, respecting the concurrency limit.

        Parameters
        ----------
        jobs : list of (experiment_name, sbatch_path) tuples

        Returns
        -------
        dict with keys 'completed', 'failed', each a list of experiment names.
        """
        total = len(jobs)
        logger.info(
            "Scheduler starting: %d jobs, max %d concurrent, polling every %ds",
            total, self.max_concurrent, self.poll_interval,
        )

        if self.dry_run:
            logger.info("DRY RUN — no jobs will be submitted.")
            for name, path in jobs:
                logger.info("  [dry-run] Would submit: %s → %s", name, path)
            return {"completed": [], "failed": []}

        self.pending.extend(jobs)
        self._fill_slots()

        while self.active_jobs or self.pending:
            time.sleep(self.poll_interval)
            self._poll_and_reap()
            self._fill_slots()

            # Progress report
            n_done = len(self.completed) + len(self.failed)
            n_active = len(self.active_jobs)
            n_pending = len(self.pending)
            logger.info(
                "Progress: %d/%d done | %d active | %d pending | %d failed",
                n_done, total, n_active, n_pending, len(self.failed),
            )

        self._print_summary()

        return {
            "completed": [r.experiment_name for r in self.completed],
            "failed": [r.experiment_name for r in self.failed],
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _fill_slots(self) -> None:
        """Submit pending jobs until we hit the concurrency limit."""
        while self.pending and len(self.active_jobs) < self.max_concurrent:
            name, sbatch_path = self.pending.popleft()
            job_id = self._sbatch_submit(sbatch_path)
            if job_id:
                record = JobRecord(
                    job_id=job_id,
                    experiment_name=name,
                    sbatch_path=sbatch_path,
                    status="RUNNING",
                )
                self.active_jobs[job_id] = record
                logger.info("Submitted %s as job %s", name, job_id)
            else:
                logger.error("Failed to submit %s", name)
                self.failed.append(
                    JobRecord(job_id="", experiment_name=name,
                              sbatch_path=sbatch_path, status="SUBMIT_FAILED")
                )

    def _poll_and_reap(self) -> None:
        """Check which active jobs have finished and reap them."""
        if not self.active_jobs:
            return

        running_ids = self._get_running_job_ids()

        for job_id in list(self.active_jobs.keys()):
            if job_id not in running_ids:
                record = self.active_jobs.pop(job_id)
                exit_status = self._get_job_exit_status(job_id)

                if exit_status == "COMPLETED":
                    record.status = "COMPLETED"
                    self.completed.append(record)
                    logger.info(
                        "✓ %s (job %s) completed successfully",
                        record.experiment_name, job_id,
                    )
                else:
                    record.status = exit_status or "FAILED"
                    self.failed.append(record)
                    logger.warning(
                        "✗ %s (job %s) ended with status: %s",
                        record.experiment_name, job_id, record.status,
                    )

    def _sbatch_submit(self, sbatch_path: str) -> Optional[str]:
        """Submit an sbatch script and return the job ID, or None on failure."""
        try:
            result = subprocess.run(
                ["sbatch", sbatch_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.error("sbatch error: %s", result.stderr.strip())
                return None
            # Parse "Submitted batch job 12345"
            return result.stdout.strip().split()[-1]
        except Exception as e:
            logger.error("sbatch exception: %s", e)
            return None

    def _get_running_job_ids(self) -> set[str]:
        """Query squeue for all job IDs owned by the current user."""
        try:
            result = subprocess.run(
                ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return set(result.stdout.strip().split())
        except Exception as e:
            logger.warning("squeue failed: %s — assuming all jobs running", e)
            return set(self.active_jobs.keys())

    def _get_job_exit_status(self, job_id: str) -> str:
        """Query sacct for the final status of a completed job."""
        try:
            result = subprocess.run(
                [
                    "sacct", "-j", job_id,
                    "--format=State", "--noheader", "--parsable2",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            states = [
                s.strip()
                for s in result.stdout.strip().split("\n")
                if s.strip()
            ]
            # sacct can return multiple lines (job + job steps);
            # the first line is the overall job status
            return states[0] if states else "UNKNOWN"
        except Exception as e:
            logger.warning("sacct failed for job %s: %s", job_id, e)
            return "UNKNOWN"

    def _print_summary(self) -> None:
        """Print a final summary table."""
        total = len(self.completed) + len(self.failed)
        logger.info("")
        logger.info("=" * 60)
        logger.info("SCHEDULER SUMMARY")
        logger.info("=" * 60)
        logger.info("  Total jobs:     %d", total)
        logger.info("  Completed:      %d", len(self.completed))
        logger.info("  Failed:         %d", len(self.failed))
        logger.info("")

        if self.completed:
            logger.info("  ✓ Completed:")
            for r in self.completed:
                logger.info("      %s (job %s)", r.experiment_name, r.job_id)

        if self.failed:
            logger.info("  ✗ Failed:")
            for r in self.failed:
                logger.info(
                    "      %s (job %s, status: %s)",
                    r.experiment_name, r.job_id, r.status,
                )

        logger.info("=" * 60)
