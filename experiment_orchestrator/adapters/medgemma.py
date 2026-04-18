"""
medgemma.py — Adapter for MedGemma model variants.

MedGemma uses 2D PNG images via PIL, AutoProcessor, and Pydantic JSON-
constrained generation.  Supports three variants:
  - multi_slice:  3 axial/coronal/sagittal PNGs per patient
  - single_slice: 1 axial PNG per patient
  - blank:        a single blacked-out PNG (control experiment)
"""
from __future__ import annotations

import posixpath

from experiment_orchestrator.adapters.base import ModelAdapter
from experiment_orchestrator.config_resolver import ResolvedJob


class MedGemmaAdapter(ModelAdapter):
    """Builds CLI commands for MedGemma testing scripts."""

    VARIANT_SCRIPTS = {
        "multi_slice": "testing_scripts/QA_testing_medgemma_multi_slice.py",
        "single_slice": "testing_scripts/QA_testing_medgemma_single_slice.py",
        "blank": "testing_scripts/QA_testing_medgemma_blank.py",
    }

    def build_command(self, job: ResolvedJob, project_root: str) -> str:
        script_rel = self.VARIANT_SCRIPTS.get(job.variant)
        if script_rel is None:
            raise ValueError(
                f"Unknown MedGemma variant '{job.variant}'. "
                f"Available: {list(self.VARIANT_SCRIPTS.keys())}"
            )

        script = posixpath.join(project_root, script_rel)
        args = [
            f"--qa_path {job.qa_path}",
            f"--output_path {job.output_path}",
            f"--model_path {job.model_path}",
        ]

        if job.variant == "blank":
            args.append(f"--image_path {job.image_path}")
        else:
            args.append(f"--image_dir {job.image_dir}")

        return f"{script} {' '.join(args)}"

    def validate(self, job: ResolvedJob) -> None:
        # Check variant is known
        known = set(self.VARIANT_SCRIPTS.keys())
        if job.variant not in known:
            raise ValueError(
                f"MedGemma variant '{job.variant}' not in {known}"
            )

        if job.variant == "blank":
            if not job.image_path:
                raise ValueError(
                    f"MedGemma blank test '{job.job_name}' requires 'image_path'"
                )
        else:
            if not job.image_dir:
                raise ValueError(
                    f"MedGemma test '{job.job_name}' requires 'image_dir'"
                )
