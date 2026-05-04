"""
openai_versa.py — Adapter for GPT-5+ vision via UCSF Versa / Mulesoft Azure OpenAI.

A single script (QA_testing_OpenAI.py) handles all variants; the variant
is conveyed through CLI flags:
  - single_slice  : --image_dir <per-patient PNG dir>
  - multi_slice   : --image_dir + multiple filenames
  - montage_slice : --image_dir + montage filename
  - blank         : --image_path <blacked-out PNG>
  - text_only     : --text_only  (no images sent at all)

Unlike local GPU models, this adapter:
  - Does not require --model_path (auth is via .env / OPENAI_API_KEY)
  - Does not need a GPU partition (API call, no local inference)
  - Supports --deployment to switch between GPT-5 / GPT-4.1 / o4-mini
"""
from __future__ import annotations

import posixpath

from experiment_orchestrator.adapters.base import ModelAdapter
from experiment_orchestrator.config_resolver import ResolvedJob


class OpenAIVersaAdapter(ModelAdapter):
    """Builds CLI commands for QA_testing_OpenAI.py (Versa/Azure GPT-5+)."""

    SCRIPT = "testing_scripts/QA_testing_OpenAI.py"

    SUPPORTED_VARIANTS = {"single_slice", "multi_slice", "montage_slice", "multi_montage_slice", "blank", "text_only"}

    # Filename to pass for each image-dir-based variant
    _IMAGE_FILENAME = {
        "single_slice":  "axial_FLAIR.png",
        "multi_slice":   "axial_FLAIR.png coronal_FLAIR.png sagittal_FLAIR.png",
        "montage_slice": "axial_slices_montage.png",
        "multi_montage_slice": "axial_slices_montage.png coronal_slices_montage.png sagittal_slices_montage.png",
    }

    def build_command(self, job: ResolvedJob, project_root: str) -> str:
        if job.variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unknown OpenAI variant '{job.variant}'. "
                f"Available: {sorted(self.SUPPORTED_VARIANTS)}"
            )

        script = posixpath.join(project_root, self.SCRIPT)
        args = [
            f"--qa_path {job.qa_path}",
            f"--output_path {job.output_path}",
        ]

        if job.variant == "text_only":
            # No image arguments — pass the flag so the script skips image loading
            args.append("--text_only")
        elif job.variant == "blank":
            args.append(f"--image_path {job.image_path}")
        else:
            args.append(f"--image_dir {job.image_dir}")
            args.append(f"--image_filename {self._IMAGE_FILENAME[job.variant]}")

        # Optional deployment override (e.g. gpt-4.1-2025-04-14)
        if getattr(job, "deployment", None):
            args.append(f"--deployment {job.deployment}")

        # Parallel request batching (threading)
        args.append(f"--batch_size {job.batch_size}")

        return f"{script} {' '.join(args)}"

    def validate(self, job: ResolvedJob) -> None:
        if job.variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"OpenAI variant '{job.variant}' not in {sorted(self.SUPPORTED_VARIANTS)}"
            )

        if job.variant == "text_only":
            # No image required — nothing to validate
            pass
        elif job.variant == "blank":
            if not job.image_path:
                raise ValueError(
                    f"OpenAI blank test '{job.job_name}' requires 'image_path'"
                )
        else:
            if not job.image_dir:
                raise ValueError(
                    f"OpenAI test '{job.job_name}' requires 'image_dir'"
                )
