"""
config_schema.py — Pydantic models for experiment.json validation.

Defines the hierarchical schema: global → models → tests.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class GlobalConfig(BaseModel):
    """Top-level settings shared across all experiments."""

    max_concurrent_jobs: int = Field(
        default=4,
        description="Maximum number of SLURM jobs running simultaneously.",
    )
    runs_per_experiment: int = Field(
        default=3,
        description="Number of times each test is repeated for averaging.",
    )

    partition: str = "gpu"
    gpus_per_node: int = 1
    cpus_per_task: int = 16
    mem: str = "20G"
    time: str = "2-00:00:00"
    mail_user: str = ""
    mail_type: str = "ALL"
    log_dir: str = "/home/remote/%u/logs"
    qa_path: str = Field(
        description="Default ground-truth QA dataset CSV path.",
    )
    output_base: str = Field(
        description="Root directory for experiment output CSVs.",
    )
    project_root: str = Field(
        description="Absolute path to the QA-BrainTumor-VLM-UCSF- repo on the cluster.",
    )
    poll_interval_seconds: int = Field(
        default=30,
        description="How often (seconds) the scheduler checks for completed jobs.",
    )
    batch_size: int = Field(
        default=4,
        description="Default number of QA rows per model.generate() call across all models.",
    )

    def slurm_defaults(self) -> dict:
        """Return the SLURM-relevant defaults as a flat dict for merging."""
        return {
            "partition": self.partition,
            "gpus_per_node": self.gpus_per_node,
            "cpus_per_task": self.cpus_per_task,
            "mem": self.mem,
            "time": self.time,
            "mail_user": self.mail_user,
            "mail_type": self.mail_type,
            "log_dir": self.log_dir,
        }


class EnvironmentConfig(BaseModel):
    """Defines how a conda env / venv / static python path is activated in SLURM."""

    type: Literal["conda_dynamic", "static_path", "venv"]
    env_name: Optional[str] = None          # for conda_dynamic
    python_path: Optional[str] = None       # for static_path
    ld_library_path: Optional[str] = None   # for static_path
    activate_path: Optional[str] = None     # for venv


class TestConfig(BaseModel):
    """A single test variant within a model definition."""

    name: str = Field(description="Unique name for this test (e.g., 'multi_slice').")
    variant: str = Field(description="Adapter-understood variant key.")
    output_path: Optional[str] = Field(default=None, description="Where results CSV is written.")

    # Optional overrides (inherit from model → global if unset)
    qa_path: Optional[str] = None
    image_path: Optional[str] = None
    image_dir: Optional[str] = None

    runs_per_experiment: Optional[int] = None
    slurm_overrides: dict = {}
    enabled: bool = True
    shuffled: bool = False


class ModelConfig(BaseModel):
    """
    A model type definition (e.g., 'MedGemma1.5').
    Contains shared settings and a list of test variants.
    """

    model_config = {"protected_namespaces": ()}

    enabled: bool = True
    adapter: str = Field(description="Key into the adapter registry (e.g., 'medgemma').")
    environment: str = Field(description="Key into the environments dict.")
    model_path: str = Field(description="Path to model weights on the cluster.")

    # Shared defaults for all tests under this model
    image_dir: Optional[str] = None

    runs_per_experiment: Optional[int] = None
    slurm_overrides: dict = {}
    # Per-model batch_size override (inherits from global if not set)
    batch_size: Optional[int] = None

    tests: list[TestConfig]


class ExperimentSuite(BaseModel):
    """Root schema for experiment.json."""

    global_config: GlobalConfig = Field(alias="global")
    environments: dict[str, EnvironmentConfig]
    models: dict[str, ModelConfig]
