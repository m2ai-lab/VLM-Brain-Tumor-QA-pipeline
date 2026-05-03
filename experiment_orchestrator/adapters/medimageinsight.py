"""
medimageinsight.py — Adapter for MedImageInsight zero-shot classifier.

MedImageInsight is a CLIP-style contrastive embedding model. It does NOT
generate text — instead it computes cosine similarity between an image
embedding and text-label embeddings to do zero-shot classification.

The testing script parses answer options from the question, encodes both
the image and options, and picks the highest-similarity answer.
"""
from __future__ import annotations

import posixpath

from experiment_orchestrator.adapters.base import ModelAdapter
from experiment_orchestrator.config_resolver import ResolvedJob


class MedImageInsightAdapter(ModelAdapter):
    """Builds CLI commands for MedImageInsight testing scripts."""

    _IMAGE_FILENAME = {
        "single_slice":  "Axial.png",
        "montage_slice": "axial_slices_montage.png",
    }

    def build_command(self, job: ResolvedJob, project_root: str) -> str:
        script = posixpath.join(
            project_root, "testing_scripts/QA_testing_MedImageInsight.py"
        )
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

        args.append(f"--batch_size {job.batch_size}")

        return f"{script} {' '.join(args)}"

    def validate(self, job: ResolvedJob) -> None:
        if job.variant == "blank":
            if not job.image_path:
                raise ValueError(
                    f"MedImageInsight blank test '{job.job_name}' requires 'image_path'"
                )
        else:
            if not job.image_dir:
                raise ValueError(
                    f"MedImageInsight test '{job.job_name}' requires 'image_dir'"
                )
