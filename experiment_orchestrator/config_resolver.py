"""
config_resolver.py — Flatten the hierarchical experiment.json into individual jobs.

Applies the inheritance chain:
    global → model → test
for runs_per_experiment, qa_path, image_dir, and slurm params.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from experiment_orchestrator.config_schema import ExperimentSuite, ModelConfig, TestConfig

# Resolve config_utils from the project root (two levels up from this file)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config_utils import load_config


def _expand(value: Any, cfg: dict) -> Any:
    """Recursively expand {variable} placeholders in strings using config.yaml values."""
    if isinstance(value, str):
        try:
            return value.format_map(cfg)
        except (KeyError, ValueError):
            return value
    if isinstance(value, dict):
        return {k: _expand(v, cfg) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(item, cfg) for item in value]
    return value


def expand_suite_raw(raw: dict) -> dict:
    """
    Expand all {variable} placeholders in the raw experiment.json dict
    using values from config.yaml before Pydantic validation.
    """
    cfg = load_config()
    return _expand(raw, cfg)


@dataclass
class ResolvedJob:
    """A single, fully-resolved SLURM job to submit."""

    job_name: str           # e.g., "MedGemma1.5_multi_slice_run2"
    model_name: str         # e.g., "MedGemma1.5"
    test_name: str          # e.g., "multi_slice_shuffled" (for naming)
    variant: str            # e.g., "multi_slice" (for script selection)
    run_number: int         # 1-indexed
    total_runs: int         # total number of runs for this test
    adapter_name: str       # e.g., "medgemma"
    environment: str        # key into environments dict
    model_path: str
    image_dir: Optional[str]
    image_path: Optional[str]
    qa_path: str
    output_path: str        # with _runN suffix
    slurm_params: dict = field(default_factory=dict)
    batch_size: int = 4     # resolved: model override → global default
    shuffled: bool = False



def _resolve_runs(
    test: TestConfig, model: ModelConfig, global_runs: int
) -> int:
    """Pick the most specific runs_per_experiment (test > model > global)."""
    if test.runs_per_experiment is not None:
        return test.runs_per_experiment
    if model.runs_per_experiment is not None:
        return model.runs_per_experiment
    return global_runs


def _make_run_output_path(base_output_path: str, run_number: int, total_runs: int) -> str:
    """
    Insert _runN before .csv extension.
    If total_runs == 1, return the original path (no suffix needed).
    """
    if total_runs == 1:
        return base_output_path
    stem, ext = os.path.splitext(base_output_path)
    return f"{stem}_run{run_number}{ext}"


def resolve_all(suite: ExperimentSuite) -> list[ResolvedJob]:
    """
    Walk the model→test hierarchy and produce a flat list of ResolvedJobs,
    one per (model, test, run_number) combination.
    """
    jobs: list[ResolvedJob] = []

    for model_name, model_cfg in suite.models.items():
        if not model_cfg.enabled:
            continue

        for test in model_cfg.tests:
            if not test.enabled:
                continue

            # --- Resolve inherited values ---
            num_runs = _resolve_runs(
                test, model_cfg, suite.global_config.runs_per_experiment
            )
            qa_path = test.qa_path or suite.global_config.qa_path
            image_dir = test.image_dir or model_cfg.image_dir
            image_path = test.image_path  # only set for blank variants

            # Merge SLURM params: global < model < test
            slurm = {
                **suite.global_config.slurm_defaults(),
                **model_cfg.slurm_overrides,
                **test.slurm_overrides,
            }

            # Resolve batch_size: model override → global default
            effective_batch_size = (
                model_cfg.batch_size
                if model_cfg.batch_size is not None
                else suite.global_config.batch_size
            )

            # Resolve output_path: use test.output_path if set, else use default pattern
            base_output_path = test.output_path
            if not base_output_path:
                base_output_path = os.path.join(
                    suite.global_config.output_base,
                    model_name,
                    f"{test.name}_results.csv"
                )

            for run in range(1, num_runs + 1):
                output_path = _make_run_output_path(base_output_path, run, num_runs)

                jobs.append(
                    ResolvedJob(
                        job_name=f"{model_name}_{test.name}_run{run}",
                        model_name=model_name,
                        test_name=test.name,
                        variant=test.variant,
                        run_number=run,
                        total_runs=num_runs,
                        adapter_name=model_cfg.adapter,
                        environment=model_cfg.environment,
                        model_path=model_cfg.model_path,
                        image_dir=image_dir,
                        image_path=image_path,
                        qa_path=qa_path,
                        output_path=output_path,
                        slurm_params=slurm,
                        batch_size=effective_batch_size,
                        shuffled=test.shuffled,
                    )
                )

    return jobs
