"""
medfusion.py — Adapter for MedicalGroundedFusionVLM.

MedFusion uses its own testing script located in the LG-Fusion repo and
requires a dedicated venv. It typically needs higher memory (128G).
"""
from __future__ import annotations

import posixpath

from experiment_orchestrator.adapters.base import ModelAdapter
from experiment_orchestrator.config_resolver import ResolvedJob


class MedFusionAdapter(ModelAdapter):
    """Builds CLI commands for MedFusion (GroundedFusionVLM) testing scripts."""

    def build_command(self, job: ResolvedJob, project_root: str, overwrite: bool = False) -> str:
        # MedFusion has its own script outside the main repo
        script = posixpath.join(
            job.model_path, "testing/RunMedicalGroundedFusionVLM.py"
        )
        ov = " --overwrite" if overwrite else ""
        sh = " --shuffled" if job.shuffled else ""
        return (
            f"{script}"
            f" --qa_path {job.qa_path}"
            f" --output_path {job.output_path}"
            f"{ov}"
            f"{sh}"
        )

    def validate(self, job: ResolvedJob) -> None:
        if not job.model_path:
            raise ValueError(
                f"MedFusion test '{job.job_name}' requires 'model_path'"
            )
