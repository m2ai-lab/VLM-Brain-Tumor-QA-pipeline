"""
lingshu.py — Adapter for Lingshu-32B model.

Lingshu-32B is a medical VLM that can be loaded via HuggingFace Transformers.
This adapter handles the same image-filename conventions as other models.
"""
from __future__ import annotations

import posixpath

from experiment_orchestrator.adapters.base import ModelAdapter
from experiment_orchestrator.config_resolver import ResolvedJob


class LingshuAdapter(ModelAdapter):
    """Builds CLI commands for Lingshu-32B testing scripts."""

    SCRIPT = "testing_scripts/QA_testing_lingshu.py"

    SUPPORTED_VARIANTS = {"single_slice", "montage_slice", "blank"}

    # Filename to pass for each image-dir-based variant
    _IMAGE_FILENAME = {
        "single_slice":  "Axial.png",
        "montage_slice": "axial_slices_montage.png",
    }

    def build_command(self, job: ResolvedJob, project_root: str) -> str:
        if job.variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unknown Lingshu variant '{job.variant}'. "
                f"Available: {sorted(self.SUPPORTED_VARIANTS)}"
            )

        script = posixpath.join(project_root, self.SCRIPT)
        args = [
            f"--qa_path {job.qa_path}",
            f"--output_path {job.output_path}",
            f"--model_path {job.model_path}",
        ]

        if job.variant == "blank":
            args.append(f"--image_path {job.image_path}")
        else:
            args.append(f"--image_dir {job.image_dir}")
            if job.variant in self._IMAGE_FILENAME:
                args.append(f"--image_filename {self._IMAGE_FILENAME[job.variant]}")

        return f"{script} {' '.join(args)}"

    def validate(self, job: ResolvedJob) -> None:
        if job.variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Lingshu variant '{job.variant}' not in {sorted(self.SUPPORTED_VARIANTS)}"
            )

        if job.variant == "blank":
            if not job.image_path:
                raise ValueError(
                    f"Lingshu blank test '{job.job_name}' requires 'image_path'"
                )
        else:
            if not job.image_dir:
                raise ValueError(
                    f"Lingshu test '{job.job_name}' requires 'image_dir'"
                )
        
        if not job.model_path:
            raise ValueError(
                f"Lingshu test '{job.job_name}' requires 'model_path' (HuggingFace ID or local path)"
            )
