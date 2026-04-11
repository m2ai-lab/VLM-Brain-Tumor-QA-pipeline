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
from pydantic import BaseModel, Field, ValidationError, field_validator
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
        choices=[1, 2],
        default=[1, 2],
        help="Which stages to run (1=accuracy, 2=aggregate).",
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
def _normalize_question(text) -> str | None:
    """Normalize question text for robust matching across CSVs."""
    if pd.isna(text):
        return None
    s = str(text).strip().lower()
    if not s:
        return None
    s = re.sub(r"\s+", " ", s)  # collapse repeated whitespace
    return s


def _find_results_question_col(df: pd.DataFrame) -> str:
    """Find the question text column in results_df."""
    candidates = [
        "Question",
        "question",
        "question_text",
        "prompt",
        "Prompt",
        "query",
        "text",
    ]
    by_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in by_lower:
            return by_lower[c.lower()]
    raise ValueError(
        "Could not find a question column in results_df. "
        f"Columns seen: {list(df.columns)}"
    )


def stage1_eval_accuracy(args: argparse.Namespace) -> dict[str, float]:
    """
    For every *_results*.csv found under qa_path:
      - match rows by normalized QUESTION TEXT 
      - ignore predictions whose question does not exist in answer_df
      - save a *_wrongs.csv alongside the results file
      - return a {test_name: accuracy} dict and write it to metrics_dir/Evals.json
    """
    print("\n" + "=" * 60)
    print("STAGE 1 — Evaluating per-model accuracy")
    print("=" * 60)

    answer_df = pd.read_csv(args.answer_path)
    answer_idx_col = answer_df.columns[0]

    required = {"Question", "Answer"}
    missing = required - set(answer_df.columns)
    if missing:
        raise ValueError(
            f"answer_df is missing required column(s): {sorted(missing)}. "
            f"Columns: {list(answer_df.columns)}"
        )

    answer_key = answer_df[[answer_idx_col, "Question", "Answer"]].copy()
    answer_key = answer_key.rename(columns={answer_idx_col: "Question Index"})
    answer_key["__qnorm"] = answer_key["Question"].map(_normalize_question)
    answer_key = answer_key.dropna(subset=["__qnorm"])

    if answer_key["__qnorm"].duplicated().any():
        dup_n = int(answer_key["__qnorm"].duplicated().sum())
        print(f"  [warn] answer_df has {dup_n} duplicate questions after normalization; keeping first.")
        answer_key = answer_key.drop_duplicates(subset=["__qnorm"], keep="first")

    accuracy: dict[str, float] = {}

    for test_name, read_path, write_path in iter_result_csvs(args.qa_path):
        results_df = pd.read_csv(read_path)
        print(f"  Evaluating [{test_name}]  ←  {read_path}")

        if "predicted_answer" not in results_df.columns:
            print("    [warn] missing 'predicted_answer' column; skipping file.")
            continue

        try:
            result_q_col = _find_results_question_col(results_df)
        except ValueError as e:
            print(f"    [warn] {e} Skipping file.")
            continue

        results_key = results_df[[result_q_col, "predicted_answer"]].copy()
        results_key = results_key.rename(columns={result_q_col: "Question"})
        results_key["__qnorm"] = results_key["Question"].map(_normalize_question)
        results_key = results_key.dropna(subset=["__qnorm"])

        if results_key["__qnorm"].duplicated().any():
            dup_n = int(results_key["__qnorm"].duplicated().sum())
            print(f"    [warn] results_df has {dup_n} duplicate questions after normalization; keeping first.")
            results_key = results_key.drop_duplicates(subset=["__qnorm"], keep="first")

        merged = results_key.merge(
            answer_key[["__qnorm", "Question Index", "Answer"]],
            on="__qnorm",
            how="inner",
        )

        valid_preds = len(results_key)
        matched = len(merged)
        ignored_not_in_answer = valid_preds - matched

        if matched == 0:
            print("    [warn] no overlapping questions with answer_df; accuracy set to 0.0")
            pd.DataFrame(
                columns=["Question Index", "Correct_Answer", "Predicted_Answer"]
            ).to_csv(write_path, index=False)
            accuracy[test_name] = 0.0
            continue

        is_right = merged.apply(
            lambda r: str(r["Answer"]) in str(r["predicted_answer"]),
            axis=1,
        )

        wrong_df = merged.loc[~is_right, ["Question Index", "Answer", "predicted_answer"]].copy()
        wrong_df = wrong_df.rename(
            columns={"Answer": "Correct_Answer", "predicted_answer": "Predicted_Answer"}
        )
        wrong_df.to_csv(write_path, index=False)

        total_right = int(is_right.sum())
        acc = total_right / matched
        accuracy[test_name] = acc

        print(
            f"    matched={matched}, ignored_not_in_answer={ignored_not_in_answer}, "
            f"accuracy={acc:.4f}  |  wrongs saved → {write_path}"
        )

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
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = argument_handler()
    os.makedirs(args.metrics_dir, exist_ok=True)

    if 1 in args.stages:
        stage1_eval_accuracy(args)

    if 2 in args.stages:
        stage2_aggregate_rw(args)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()