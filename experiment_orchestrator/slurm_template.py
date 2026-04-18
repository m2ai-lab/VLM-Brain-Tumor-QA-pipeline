"""
slurm_template.py — Generate .sbatch scripts from experiment config.

Produces a complete bash script for each resolved job, handling all three
environment activation patterns used in the project:
  - conda_dynamic:  grep conda env list for the named environment
  - static_path:    hardcoded python path + LD_LIBRARY_PATH
  - venv:           source activate a virtualenv
"""
from __future__ import annotations

import os
import textwrap

from experiment_orchestrator.config_resolver import ResolvedJob
from experiment_orchestrator.config_schema import EnvironmentConfig


# ─────────────────────────────────────────────────────────────────────────────
# Environment activation templates
# ─────────────────────────────────────────────────────────────────────────────

CONDA_DYNAMIC_TEMPLATE = textwrap.dedent("""\
    # DYNAMIC ENVIRONMENT DETECTION
    ENV_PATH=$(conda env list | grep -E '^{env_name}\\s' | awk '{{print $NF}}')

    if [ -z "$ENV_PATH" ]; then
        echo "ERROR: Environment '{env_name}' not found for user $USER."
        echo "Please create it using: conda create -n {env_name} python=3.10"
        exit 1
    fi

    DYNAMIC_PYTHON="${{ENV_PATH}}/bin/python"
""")

STATIC_PATH_TEMPLATE = textwrap.dedent("""\
    # STATIC PYTHON PATH
    DYNAMIC_PYTHON="{python_path}"
""")

STATIC_PATH_LD_TEMPLATE = textwrap.dedent("""\
    # Force library priority
    export LD_LIBRARY_PATH="{ld_library_path}:$LD_LIBRARY_PATH"
""")

VENV_TEMPLATE = textwrap.dedent("""\
    # ACTIVATE VIRTUALENV
    source {activate_path}
    DYNAMIC_PYTHON="python"
""")


# ─────────────────────────────────────────────────────────────────────────────
# Main sbatch template
# ─────────────────────────────────────────────────────────────────────────────

SBATCH_TEMPLATE = textwrap.dedent("""\
    #!/bin/bash
    #SBATCH --job-name={job_name}
    #SBATCH --time={time}
    #SBATCH --partition={partition}
    #SBATCH --ntasks-per-node=1
    #SBATCH --nodes=1
    #SBATCH --gres=gpu:{gpus_per_node}
    #SBATCH --cpus-per-task={cpus_per_task}
    #SBATCH --mem={mem}
    #SBATCH --output={log_dir}/%j_%x.out
    #SBATCH --error={log_dir}/%j_%x.err
    {mail_lines}

    # ── Environment Isolation ─────────────────────────────────────────────
    module load miniforge3
    export PYTHONNOUSERSITE=1
    unset PYTHONPATH

    {env_block}

    echo "------------------------------------------------------------"
    echo "Job ID:        $SLURM_JOB_ID"
    echo "Job Name:      {job_name}"
    echo "User:          $USER"
    echo "Using Python:  $DYNAMIC_PYTHON"

    echo "Run:           {run_number}/{total_runs}"
    echo "------------------------------------------------------------"

    # ── Execute ───────────────────────────────────────────────────────────
    $DYNAMIC_PYTHON {command}

    echo "------------------------------------------------------------"
    echo "Job finished at $(date)"
""")


def _build_env_block(env_cfg: EnvironmentConfig) -> str:
    """Produce the bash block that sets up the python environment."""
    if env_cfg.type == "conda_dynamic":
        return CONDA_DYNAMIC_TEMPLATE.format(env_name=env_cfg.env_name)

    if env_cfg.type == "static_path":
        block = STATIC_PATH_TEMPLATE.format(python_path=env_cfg.python_path)
        if env_cfg.ld_library_path:
            block += STATIC_PATH_LD_TEMPLATE.format(
                ld_library_path=env_cfg.ld_library_path
            )
        return block

    if env_cfg.type == "venv":
        return VENV_TEMPLATE.format(activate_path=env_cfg.activate_path)

    raise ValueError(f"Unknown environment type: {env_cfg.type}")


def _build_mail_lines(slurm_params: dict) -> str:
    """Build SBATCH mail directives, or empty string if no mail_user."""
    mail_user = slurm_params.get("mail_user", "")
    if not mail_user:
        return ""
    mail_type = slurm_params.get("mail_type", "ALL")
    return (
        f"#SBATCH --mail-user={mail_user}\n"
        f"#SBATCH --mail-type={mail_type}"
    )


def generate_sbatch(
    job: ResolvedJob,
    env_cfg: EnvironmentConfig,
    command: str,
) -> str:
    """
    Produce the full sbatch script content for a resolved job.

    Parameters
    ----------
    job : ResolvedJob
        The fully-resolved experiment job.
    env_cfg : EnvironmentConfig
        Environment activation config for this job.
    command : str
        The script + args to run (produced by the model adapter).

    Returns
    -------
    str
        Complete bash script content ready to write to a .sbatch file.
    """
    env_block = _build_env_block(env_cfg)
    mail_lines = _build_mail_lines(job.slurm_params)

    return SBATCH_TEMPLATE.format(
        job_name=job.job_name,
        time=job.slurm_params.get("time", "2-00:00:00"),
        partition=job.slurm_params.get("partition", "gpu"),
        gpus_per_node=job.slurm_params.get("gpus_per_node", 1),
        cpus_per_task=job.slurm_params.get("cpus_per_task", 16),
        mem=job.slurm_params.get("mem", "20G"),
        log_dir=job.slurm_params.get("log_dir", "/home/remote/%u/logs"),
        mail_lines=mail_lines,
        env_block=env_block,
        command=command,

        run_number=job.run_number,
        total_runs=job.total_runs,
    )


def write_sbatch_file(
    job: ResolvedJob,
    env_cfg: EnvironmentConfig,
    command: str,
    output_dir: str,
) -> str:
    """
    Generate and write an sbatch script to disk.

    Returns the path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    content = generate_sbatch(job, env_cfg, command)
    filename = f"{job.job_name}.sbatch"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", newline="\n", encoding="utf-8") as f:
        f.write(content)

    return filepath
