# UCSF-PDGM-VQA: Visual Question Answering dataset for brain tumor MRI interpretation

This repository is the official implementation of [UCSF-PDGM-VQA: Visual Question Answering dataset for brain tumor MRI interpretation](). 

---

## Requirements

To install requirements and set up the environment, run:

```bash
# 1. Clone and Setup
git clone <repo_url>
cd VLM_BrainTumor_QA

# 2. Run interactive setup
# Configures config.yaml, creates envs, and downloads datasets/models
bash setup.sh
```

> [!IMPORTANT]
> **Configuration**: Verify `nifti_root`, `qa_path`, `human_qa_path`, and `reshuffled_qa_path` are correctly defined in `config.yaml` and that the corresponding data is present.

---

## Training (Running Experiments)

The orchestrator manages all experiments via a central `experiment.json` config. Settings inherit from **global → model → test**.

### Submit to SLURM
```bash
sbatch slurm_scripts/sbatch_run_orchestration
```

### Local Execution
If running on a machine without SLURM (e.g., local GPU station):
```bash
# Sequential
python -m experiment_orchestrator.run_local --model GPT5Mini

# Parallelize API-bound tasks (e.g. OpenAI)
python -m experiment_orchestrator.run_local --model GPT5Mini --jobs 4
```

### Basic CLI Filtering
```bash
# Preview matching jobs
python experiment_orchestrator/run_experiments.py --list

# Filter by model or test name
python experiment_orchestrator/run_experiments.py --model MedGemma1.5 --test mri_single
```

### Run Specs (YAML)
For complex batches, define a `run_spec.yaml` and execute:
```bash
python experiment_orchestrator/run_experiments.py --run-spec run_spec.yaml
```

---

## Evaluation

To evaluate model performance on the benchmark, run:

```bash
python evaluation_scripts/eval_answers.py --stages 1 2 3
```

The evaluation pipeline computes accuracy in three stages:
1. **Stage 1**: Matches results to ground truth and computes per-test accuracy.
2. **Stage 2**: Aggregates rights/wrongs to find the most challenging questions.
3. **Stage 3**: Averages accuracy across multiple runs (`_runN` suffix).

---

## Pre-trained Models

Pre-trained model paths and adapter configurations are managed via `experiment.json`. 
- **Adapters**: Adding a new model requires creating a `ModelAdapter` in `experiment_orchestrator/adapters/`. See `experiment_orchestrator/adapters/base.py` for implementation details.

---

## Results

Our models achieve the following performance on the UCSF-PDGM-VQA benchmark:

| Model Name | Slice Accuracy | Human Subset | Montage | Reshuffled | Text-Only |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-8B**                      |     - |     - |     - | 47.73 | 52.57 |
| **LLaVA-Med-1.5**                  | 59.14 | 59.60 | 59.05 | 40.55 |     - |
| **MedImageInsight**                | 20.77 | 29.69 | 18.68 | 28.01 |     - |
| **Med3DVLM-Qwen-2.5-7B - 3D**      | 26.91 |     - | 42.42 | 38.55 |     - |
| **Lingshu-32B**                    | 61.40 | 63.20 | 66.04 | 52.17 |     - |
| **MedGemma-1.5-4B - Single Slice** | 55.37 | 51.69 | 43.38 | 43.38 |     - |
| **MedGemma-1.5-4B - Multi Slice**  | 63.57 | 59.08 |     - | 51.16 |     - |
| **GPT5-mini - Single Slice**       | 34.70 | 38.59 | 32.39 | 19.87 | 18.57 |
| **GPT5-mini - Multi Slice**        | 23.67 | 27.45 |     - | 22.19 |     - |

---

## Technical Notes

- **Standardized Orientation**: All extraction scripts automatically reorient NIfTI volumes to **RAS** (Right-Anterior-Superior) canonical space.
- **Reshuffling**: Use `data_pipeline/dataset_reshuffle.py` to create biased-controlled datasets (randomized option order).
- **Human Baseline**: A Tkinter GUI is available at `testing_scripts/human_testing/human_qa_gui.py` for clinician benchmarking.

---

## Contributing

This project is licensed under the **MIT License**. Contributions are welcome! Please describe your changes and ensure they align with the project's goal of benchmarking VLMs on clinical brain tumor MRI.
