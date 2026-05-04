"""
qwen.py — Adapter for Qwen2.5-7B (Text-only).

Qwen handles text-only VQA via its own testing script.
"""
from __future__ import annotations

import posixpath

from experiment_orchestrator.adapters.base import ModelAdapter
from experiment_orchestrator.config_resolver import ResolvedJob


class QwenAdapter(ModelAdapter):
    """Builds CLI commands for Qwen testing scripts."""

    def build_command(self, job: ResolvedJob, project_root: str, overwrite: bool = False) -> str:
        ov = " --overwrite" if overwrite else ""
        if job.variant == "text_only":
            script = posixpath.join(
                project_root, "testing_scripts/QA_testing_Qwen.py"
            )
            return (
                f"{script}"
                f" --qa_path {job.qa_path}"
                f" --output_path {job.output_path}"
                f" --model_path {job.model_path}"
                f" --batch_size {job.batch_size}"
                f"{ov}"
            )
        else:
            # Add other Qwen variants here if needed
            raise ValueError(f"Qwen variant '{job.variant}' not supported yet.")

    def validate(self, job: ResolvedJob) -> None:
        if not job.model_path:
            raise ValueError(
                f"Qwen test '{job.job_name}' requires 'model_path'"
            )
