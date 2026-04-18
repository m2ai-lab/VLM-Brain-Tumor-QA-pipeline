"""
med3dvlm.py — Adapter for Med3DVLM model variants.

Med3DVLM uses 3D NIfTI volumes loaded via SimpleITK, resized with trilinear
interpolation to (128, 256, 256), and injected as <im_patch> tokens.
Supports two variants:
  - full_nifti: per-patient FLAIR NIfTI from a directory
  - blank:      a single blacked-out NIfTI (control experiment)
"""
from __future__ import annotations

import posixpath

from experiment_orchestrator.adapters.base import ModelAdapter
from experiment_orchestrator.config_resolver import ResolvedJob


class Med3DVLMAdapter(ModelAdapter):
    """Builds CLI commands for Med3DVLM testing scripts."""

    def build_command(self, job: ResolvedJob, project_root: str) -> str:
        if job.variant == "blank":
            script = posixpath.join(
                project_root, "testing_scripts/QA_testing_Med3DVLM_blank.py"
            )
            return (
                f"{script}"
                f" --qa_path {job.qa_path}"
                f" --output_path {job.output_path}"
                f" --model_path {job.model_path}"
                f" --image_path {job.image_path}"
                f" --temperature {job.temperature}"
            )
        else:
            script = posixpath.join(
                project_root, "testing_scripts/QA_testing_Med3DVLM.py"
            )
            return (
                f"{script}"
                f" --qa_path {job.qa_path}"
                f" --output_path {job.output_path}"
                f" --model_path {job.model_path}"
                f" --image_dir {job.image_dir}"
                f" --temperature {job.temperature}"
            )

    def validate(self, job: ResolvedJob) -> None:
        if job.variant == "blank":
            if not job.image_path:
                raise ValueError(
                    f"Med3DVLM blank test '{job.job_name}' requires 'image_path'"
                )
        else:
            if not job.image_dir:
                raise ValueError(
                    f"Med3DVLM test '{job.job_name}' requires 'image_dir'"
                )
