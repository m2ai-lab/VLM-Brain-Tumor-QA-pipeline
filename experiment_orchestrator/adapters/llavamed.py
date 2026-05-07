"""
llavamed.py — Adapter for LLaVA-Med.

LLaVA-Med is launched via a Python wrapper script that converts QA CSVs
into JSONL, invokes the model natively, and parses JSONL back to CSV.
"""
from __future__ import annotations

import posixpath

from experiment_orchestrator.adapters.base import ModelAdapter
from experiment_orchestrator.config_resolver import ResolvedJob


class LLaVaMedAdapter(ModelAdapter):
    """Builds CLI commands for LLaVA-Med testing scripts."""

    _IMAGE_FILENAME = {
        "single_slice":  "axial_FLAIR.png",
        "montage_slice": "axial_slices_montage.png",
    }

    def build_command(self, job: ResolvedJob, project_root: str, overwrite: bool = False) -> str:
        script = posixpath.join(
            project_root, "testing_scripts/QA_testing_llava_med.py"
        )
        args = [
            f"--qa_path {job.qa_path}",
            f"--output_path {job.output_path}",
            f"--model_path {job.model_path}",
        ]

        if job.variant == "blank":
            args.append(f"--image_path {job.image_path}")
            # image_dir might be referenced in wrapper still, so we provide something safe
            if job.image_dir:
                 args.append(f"--image_dir {job.image_dir}")
        else:
            args.append(f"--image_dir {job.image_dir}")
            if job.variant in self._IMAGE_FILENAME:
                args.append(f"--image_filename {self._IMAGE_FILENAME[job.variant]}")

        if overwrite:
            args.append("--overwrite")
        if job.shuffled:
            args.append("--shuffled")

        return f"{script} {' '.join(args)}"

    def validate(self, job: ResolvedJob) -> None:
        if job.variant == "blank":
            if not job.image_path:
                raise ValueError(
                    f"LLaVA-Med blank test '{job.job_name}' requires 'image_path'"
                )
        else:
            if not job.image_dir:
                raise ValueError(
                    f"LLaVA-Med test '{job.job_name}' requires 'image_dir'"
                )
        
        if not job.model_path:
            raise ValueError(
                f"LLaVA-Med test '{job.job_name}' requires 'model_path' pointing to weights"
            )
