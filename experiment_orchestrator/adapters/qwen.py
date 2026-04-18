"""
qwen.py — Adapter for Qwen text-only model.

Qwen is a pure language model — no image inputs. It uses
AutoModelForCausalLM with a system-prompt-enforced JSON output format.
"""
from __future__ import annotations

import posixpath

from experiment_orchestrator.adapters.base import ModelAdapter
from experiment_orchestrator.config_resolver import ResolvedJob


class QwenAdapter(ModelAdapter):
    """Builds CLI commands for Qwen testing scripts."""

    def build_command(self, job: ResolvedJob, project_root: str) -> str:
        script = posixpath.join(
            project_root, "testing_scripts/QA_testing_Qwen.py"
        )
        return (
            f"{script}"
            f" --qa_path {job.qa_path}"
            f" --output_path {job.output_path}"
            f" --model_path {job.model_path}"
        )

    def validate(self, job: ResolvedJob) -> None:
        # Qwen is text-only, no image requirements
        if not job.model_path:
            raise ValueError(
                f"Qwen test '{job.job_name}' requires 'model_path'"
            )
