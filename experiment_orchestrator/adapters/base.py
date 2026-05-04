"""
base.py — Abstract base class for model adapters.

Each model type (MedGemma, Med3DVLM, Qwen, etc.) subclasses this to define
how its testing script is invoked on the cluster.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from experiment_orchestrator.config_resolver import ResolvedJob


class ModelAdapter(ABC):
    """Interface that every model adapter must implement."""

    @abstractmethod
    def build_command(self, job: ResolvedJob, project_root: str, overwrite: bool = False) -> str:
        """
        Return the full command line (script path + CLI args).

        The python interpreter prefix is NOT included here — the slurm
        template prepends it based on the environment config.
        """
        ...

    @abstractmethod
    def validate(self, job: ResolvedJob) -> None:
        """
        Validate that the ResolvedJob has all required fields for this adapter.
        Raise ValueError with a descriptive message if not.
        """
        ...
