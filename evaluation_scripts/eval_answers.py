"""
eval_pipeline.py — Unified VLM Brain-MRI Evaluation Pipeline

Stages:
  1  eval_accuracy       Compute per-model accuracy and save per-model wrongs CSVs
  2  aggregate_rw        Aggregate rights/wrongs histograms across all models + plots
  3  question_analysis   Qwen-powered contrastive analysis (global + per-model)

Run all stages:
  python eval_pipeline.py

Run specific stages:
  python eval_pipeline.py --stages 1 2
  python eval_pipeline.py --stages 3
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT (used in Stage 3)
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert AI behavior analyst. Your task is to analyze sets of questions that an AI
model answered CORRECTLY versus INCORRECTLY.
You must contrast the two sets and identify specific, actionable patterns that explain why the
model fails on certain questions. Look for differences in:
1. Question complexity (e.g., multi-step reasoning vs. factual recall)
2. Structural formats (e.g., true/false, open-ended, medical jargon)
3. Specific keywords or themes unique to the failures.

Your response MUST be a strictly valid JSON object using the exact keys below.
Do not nest the JSON inside any other objects. Do not include any markdown formatting.

{
  "success_patterns":     "Themes or structural patterns common ONLY in questions the model got right.",
  "failure_patterns":     "Themes or structural patterns common ONLY in questions the model got wrong.",
  "key_differences":      "Primary differences in wording, complexity, or subject matter between sets.",
  "insightful_conclusion":"A definitive statement on the model's blind spots and why it is failing."
}
"""


# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT HANDLER
# ──────────────────────────────────────────────────────────────────────────────
def argument_handler() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified VLM Brain-MRI Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Shared paths ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--qa_path",
        default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs",
        help="Root directory that contains all model result CSVs.",
    )
    parser.add_argument(
        "--answer_path",
        default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/UCSF_PDGM_QAPairs_Sample.csv",
        help="Master ground-truth Q&A CSV.",
    )
    parser.add_argument(
        "--metrics_dir",
        default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/metrics",
        help="Directory where all output files are written.",
    )

    # ── Stage selection ───────────────────────────────────────────────────────
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        choices=[1, 2, 3],
        default=[1, 2, 3],
        help="Which stages to run (1=accuracy, 2=aggregate, 3=analysis).",
    )

    # ── Stage 2: filtering ────────────────────────────────────────────────────
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=["qwen", "blank"],
        help="Exclude wrongs CSVs whose path contains any of these strings (case-insensitive).",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=[],
        help=(
            "If non-empty, only wrongs CSVs whose full path appears in this list are used. "
            "Leave empty to include everything not excluded."
        ),
    )

    # ── Stage 3: model ────────────────────────────────────────────────────────
    parser.add_argument(
        "--model_path",
        default="/scratch/group/CX000019_DS1/vlm-brain-mri/Qwen2.5-7B-Instruct",
        help="Local path (or HF hub name) of the Qwen model used for Stage 3.",
    )

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def iter_result_csvs(qa_path: str):
    """
    Walk qa_path and yield (test_name, read_path, write_path) for every
    CSV whose name contains 'result' (case-insensitive).
    write_path is the sibling file that Stage 1 will create for wrongs.
    """
    for root, _, files in os.walk(qa_path):
        for file in files:
            if file.lower().endswith(".csv") and "result" in file.lower():
                stem = file.lower().split(".")[0].replace("_results", "")
                test_name = f"{root.split('/')[-1]}_{stem}"
                read_path = os.path.join(root, file)
                write_path = os.path.join(root, f"{test_name}_wrongs.csv")
                yield test_name, read_path, write_path


def iter_wrongs_csvs(qa_path: str, exclude: list[str], include: list[str]):
    """
    Walk qa_path and yield wrongs CSV paths, applying include/exclude filters.
    """
    exclude_lower = [e.lower() for e in exclude]
    for root, _, files in os.walk(qa_path):
        if any(ex in root.lower() for ex in exclude_lower):
            continue
        for file in files:
            if not (file.lower().endswith(".csv") and "wrongs" in file.lower()):
                continue
            if any(ex in file.lower() for ex in exclude_lower):
                continue
            full_path = os.path.join(root, file)
            if include and full_path not in include:
                continue
            yield full_path


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Per-model accuracy
# ──────────────────────────────────────────────────────────────────────────────
def stage1_eval_accuracy(args: argparse.Namespace) -> dict[str, float]:
    """
    For every *_results*.csv found under qa_path:
      - compare predicted_answer against the master answer sheet
      - save a *_wrongs.csv alongside the results file
      - return a {test_name: accuracy} dict and write it to metrics_dir/Evals.json
    """
    print("\n" + "=" * 60)
    print("STAGE 1 — Evaluating per-model accuracy")
    print("=" * 60)

    answer_df = pd.read_csv(args.answer_path)
    question_idx_col = answer_df.iloc[:, 0]

    accuracy: dict[str, float] = {}

    for test_name, read_path, write_path in iter_result_csvs(args.qa_path):
        results_df = pd.read_csv(read_path)
        print(f"  Evaluating [{test_name}]  ←  {read_path}")

        total_right = 0
        wrong_indexes, wrong_answers, wrong_preds = [], [], []

        for idx, (ans, pred) in enumerate(
            zip(answer_df["Answer"], results_df["predicted_answer"])
        ):
            if str(ans) in str(pred):
                total_right += 1
            else:
                wrong_indexes.append(question_idx_col.iloc[idx])
                wrong_answers.append(str(ans))
                wrong_preds.append(str(pred))

        pd.DataFrame(
            {
                "Question Index": wrong_indexes,
                "Correct_Answer": wrong_answers,
                "Predicted_Answer": wrong_preds,
            }
        ).to_csv(write_path, index=False)

        acc = total_right / len(results_df)
        accuracy[test_name] = acc
        print(f"    accuracy = {acc:.4f}  |  wrongs saved → {write_path}")

    os.makedirs(args.metrics_dir, exist_ok=True)
    evals_path = os.path.join(args.metrics_dir, "Evals.json")
    with open(evals_path, "w") as f:
        json.dump(accuracy, f, indent=4)
    print(f"\n  ✓ Accuracy summary saved → {evals_path}")

    return accuracy


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Aggregate rights/wrongs across models
# ──────────────────────────────────────────────────────────────────────────────
def _plot_counter(counter: Counter, output_path: str, title: str) -> None:
    top = dict(counter.most_common(20))
    if not top:
        print(f"  [warn] No data to plot for: {title}")
        return

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=[str(k) for k in top.keys()],
        y=list(top.values()),
        hue=[str(k) for k in top.keys()],
        palette="Reds_r",
        legend=False,
    )
    plt.title(title)
    plt.xlabel("Question Index")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"    plot saved → {output_path}")


def _export_question_counts(
    counter: Counter, output_path: str, answer_df: pd.DataFrame
) -> None:
    rows = []
    for q_idx, count in counter.items():
        try:
            row = answer_df.loc[int(q_idx)]
            rows.append(
                {
                    "Count": count,
                    "Question": row["Question"],
                    "Answer": row["Answer"],
                }
            )
        except (KeyError, ValueError):
            pass
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"    data saved  → {output_path}")


def stage2_aggregate_rw(args: argparse.Namespace) -> tuple[Counter, Counter]:
    """
    Aggregate rights/wrongs histograms across all filtered wrongs CSVs,
    produce bar-chart figures and summary CSVs.
    Returns (wrongs_hist, rights_hist).
    """
    print("\n" + "=" * 60)
    print("STAGE 2 — Aggregating rights/wrongs across models")
    print("=" * 60)

    answer_df = pd.read_csv(args.answer_path)
    answer_df = answer_df.rename(columns={answer_df.columns[0]: "Question_Idx"})
    answer_df = answer_df.set_index("Question_Idx")
    all_questions = set(answer_df.index)

    wrongs_hist: Counter = Counter()
    rights_hist: Counter = Counter()

    for csv_path in iter_wrongs_csvs(args.qa_path, args.exclude, args.include):
        df = pd.read_csv(csv_path)
        if "Question Index" not in df.columns:
            continue
        print(f"  Processing {csv_path}")
        wrongs = set(df["Question Index"])
        wrongs_hist.update(wrongs)
        rights_hist.update(all_questions - wrongs)

    os.makedirs(args.metrics_dir, exist_ok=True)

    wrong_fig = os.path.join(args.metrics_dir, "Top_wrong_figure.png")
    right_fig = os.path.join(args.metrics_dir, "Top_right_figure.png")
    wrong_csv = os.path.join(args.metrics_dir, "Top_wrong.csv")
    right_csv = os.path.join(args.metrics_dir, "Top_right.csv")

    print("\n  Plotting & exporting wrongs …")
    _plot_counter(wrongs_hist, wrong_fig, "Top 20 Most Frequently Missed Questions")
    _export_question_counts(wrongs_hist, wrong_csv, answer_df)

    print("  Plotting & exporting rights …")
    _plot_counter(rights_hist, right_fig, "Top 20 Most Frequently Correct Questions")
    _export_question_counts(rights_hist, right_csv, answer_df)

    print(f"\n  ✓ Aggregation complete.")
    return wrongs_hist, rights_hist


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Qwen contrastive question analysis
# ──────────────────────────────────────────────────────────────────────────────
class QwenResponse(BaseModel):
    success_patterns: str = Field(
        description="Themes or structural patterns common ONLY in questions the model got right."
    )
    failure_patterns: str = Field(
        description="Themes or structural patterns common ONLY in questions the model got wrong."
    )
    key_differences: str = Field(
        description="Primary differences in wording, complexity, or subject matter between sets."
    )
    insightful_conclusion: str = Field(
        description="A definitive statement on the model's blind spots and why it is failing."
    )


def _clean_json_string(raw: str) -> str:
    clean = re.sub(r"```json|```", "", raw).strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    return match.group(0) if match else clean


def _query_model(model, tokenizer, user_prompt: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generated = model.generate(
        **inputs, max_new_tokens=1024, temperature=0.2, do_sample=True
    )
    new_tokens = generated[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    cleaned = _clean_json_string(raw_output)

    try:
        if not cleaned:
            raise ValueError("No JSON-like content found in model output.")
        validated = QwenResponse.model_validate_json(cleaned)
        return validated.model_dump()
    except ValidationError as e:
        print(f"    [warn] Pydantic validation error: {e}")
        return {"error": "Schema mismatch", "raw": raw_output}
    except Exception as e:
        print(f"    [warn] Parsing error: {e}")
        return {"error": str(e), "raw": raw_output}


def _build_comparison_prompts(rights_df: pd.DataFrame, wrongs_df: pd.DataFrame) -> dict:
    """Return broad + extreme contrastive prompts from two question DataFrames."""
    rights_df["Question"] = rights_df["Question"].astype(str).str.strip()
    wrongs_df["Question"] = wrongs_df["Question"].astype(str).str.strip()

    all_right = ", ".join(sorted(rights_df["Question"].dropna().unique()))
    all_wrong = ", ".join(sorted(wrongs_df["Question"].dropna().unique()))

    max_right_count = rights_df["Count"].max()
    max_wrong_count = wrongs_df["Count"].max()
    most_right = ", ".join(
        sorted(rights_df[rights_df["Count"] == max_right_count]["Question"].dropna().unique())
    )
    most_wrong = ", ".join(
        sorted(wrongs_df[wrongs_df["Count"] == max_wrong_count]["Question"].dropna().unique())
    )

    return {
        "Broad Analysis": (
            f"SUCCESSFUL QUESTIONS:\n{all_right}\n\nFAILED QUESTIONS:\n{all_wrong}"
        ),
        "Extreme Analysis": (
            f"MOST SUCCESSFUL QUESTIONS:\n{most_right}\n\nMOST FAILED QUESTIONS:\n{most_wrong}"
        ),
    }


def stage3_question_analysis(args: argparse.Namespace) -> None:
    """
    Run Qwen contrastive analysis in two passes:
      A) Global  — using the aggregate Top_right.csv / Top_wrong.csv from Stage 2
      B) Per-model — one analysis per model using its own wrongs CSV paired with the
                     global rights CSV (best proxy without re-running Stage 2 per model)

    Results are saved to metrics_dir/QA_Analysis.json.
    """
    print("\n" + "=" * 60)
    print("STAGE 3 — Qwen contrastive question analysis")
    print("=" * 60)

    global_right_csv = os.path.join(args.metrics_dir, "Top_right.csv")
    global_wrong_csv = os.path.join(args.metrics_dir, "Top_wrong.csv")

    if not os.path.exists(global_right_csv) or not os.path.exists(global_wrong_csv):
        raise FileNotFoundError(
            "Stage 3 requires the aggregate CSVs produced by Stage 2. "
            "Run Stage 2 first (or include stage 2 in --stages)."
        )

    print(f"\n  Loading Qwen model from {args.model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    final_analysis: dict = {}

    # ── A) Global analysis ────────────────────────────────────────────────────
    print("\n  [A] Global analysis across all models …")
    global_rights_df = pd.read_csv(global_right_csv)
    global_wrongs_df = pd.read_csv(global_wrong_csv)
    prompts = _build_comparison_prompts(global_rights_df, global_wrongs_df)

    final_analysis["__global__"] = {}
    for label, prompt in prompts.items():
        print(f"    Running '{label}' …")
        final_analysis["__global__"][label] = _query_model(model, tokenizer, prompt)

    # ── B) Per-model analysis ──────────────────────────────────────────────────
    print("\n  [B] Per-model analysis …")
    for wrongs_path in iter_wrongs_csvs(args.qa_path, args.exclude, args.include):
        model_name = os.path.basename(os.path.dirname(wrongs_path))
        file_name = os.path.basename(wrongs_path)

        # Derive a readable sub-level label from the filename
        sub_level = file_name.replace("_wrongs.csv", "")
        if sub_level.lower().startswith(model_name.lower() + "_"):
            sub_level = sub_level[len(model_name) + 1:]
        sub_level = sub_level or "general"

        print(f"\n    Model: {model_name} | Config: {sub_level}")
        print(f"    Wrongs CSV: {wrongs_path}")

        per_model_wrongs_df = pd.read_csv(wrongs_path)
        if "Question" not in per_model_wrongs_df.columns:
            print("    [skip] CSV has no 'Question' column.")
            continue

        prompts = _build_comparison_prompts(global_rights_df, per_model_wrongs_df)

        if model_name not in final_analysis:
            final_analysis[model_name] = {}

        final_analysis[model_name][sub_level] = {}
        for label, prompt in prompts.items():
            print(f"      Running '{label}' …")
            final_analysis[model_name][sub_level][label] = _query_model(
                model, tokenizer, prompt
            )

    os.makedirs(args.metrics_dir, exist_ok=True)
    output_path = os.path.join(args.metrics_dir, "QA_Analysis.json")
    with open(output_path, "w") as f:
        json.dump(final_analysis, f, indent=4)
    print(f"\n  ✓ Analysis saved → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = argument_handler()
    os.makedirs(args.metrics_dir, exist_ok=True)

    if 1 in args.stages:
        stage1_eval_accuracy(args)

    if 2 in args.stages:
        stage2_aggregate_rw(args)

    if 3 in args.stages:
        stage3_question_analysis(args)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()