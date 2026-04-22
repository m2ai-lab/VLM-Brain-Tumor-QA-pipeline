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
import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config
_cfg = load_config()
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
        default=_cfg.get("output_base"),
        help="Root directory that contains all model result CSVs.",
    )
    parser.add_argument(
        "--answer_dir",
        default=_cfg.get("scratch_root"),
        help="Directory containing the ground-truth Q&A CSVs.",
    )
    parser.add_argument(
        "--metrics_dir",
        default=_cfg.get("output_base", "") + "/metrics",
        help="Directory where all output files are written.",
    )

    # ── Stage selection ───────────────────────────────────────────────────────
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        choices=[1, 2, 3],
        default=[1, 2, 3],
        help="Which stages to run (1=accuracy, 2=aggregate, 3=average across runs).",
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

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_answer_csv(answer_dir: str, result_filename: str) -> str:
    """
    Pick the correct ground-truth CSV based on whether the results file
    name contains 'shuffled'.
    """
    if "shuffled" in result_filename.lower():
        return os.path.join(answer_dir, "reshuffled_finalized_ucsf_pdgm_pairs.csv")
    return os.path.join(answer_dir, "finalized_ucsf_pdgm_pairs.csv")


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


def stage1_eval_accuracy(args: argparse.Namespace) -> dict[str, float]:
    """
    For every *_results*.csv found under qa_path:
      - Dynamically choose the answer CSV (reshuffled vs. standard)
        based on whether the results filename contains 'shuffled'
      - Assume rows perfectly match answer_df order
      - check if Answer string is in predicted_answer
      - save a *_wrongs.csv alongside the results file
      - return a {test_name: accuracy} dict and write it to metrics_dir/Evals.json
    """
    print("\n" + "=" * 60)
    print("STAGE 1 — Evaluating per-model accuracy (row-by-row)")
    print("=" * 60)

    accuracy: dict[str, float] = {}

    for test_name, read_path, write_path in iter_result_csvs(args.qa_path):
        # ── pick the correct ground-truth CSV for this result file ──
        result_filename = os.path.basename(read_path)
        answer_path = _resolve_answer_csv(args.answer_dir, result_filename)
        answer_df = pd.read_csv(answer_path)
        answer_idx_col = answer_df.columns[0]
        print(f"  Evaluating [{test_name}]  ←  {read_path}")
        print(f"    answer key: {answer_path}")

        required = {"Question", "Answer"}
        missing = required - set(answer_df.columns)
        if missing:
            raise ValueError(
                f"answer_df is missing required column(s): {sorted(missing)}. "
                f"Columns: {list(answer_df.columns)}"
            )

        results_df = pd.read_csv(read_path)

        if "predicted_answer" not in results_df.columns:
            raise ValueError(
                f"No predicted_answer column found in {read_path}. "
                f"Columns: {list(results_df.columns)}"
            )

        if len(results_df) != len(answer_df):
            raise ValueError(
                f"result length ({len(results_df)}) != answer length ({len(answer_df)}). "
                f"Columns: {list(results_df.columns)}"
            )

        is_right = []
        for ans, pred in zip(answer_df["Answer"], results_df["predicted_answer"]):
            is_right.append(str(ans) in str(pred))

        wrong_df = pd.DataFrame({
            "Question Index": answer_df[answer_idx_col].iloc[:len(is_right)].values,
            "Question": answer_df["Question"].iloc[:len(is_right)].values,
            "Correct_Answer": answer_df["Answer"].iloc[:len(is_right)].values,
            "Predicted_Answer": results_df["predicted_answer"].iloc[:len(is_right)].values
        })
        wrong_df = wrong_df[~pd.Series(is_right)]
        wrong_df.to_csv(write_path, index=False)

        total_right = sum(is_right)
        acc = total_right / len(results_df) if len(results_df) > 0 else 0.0
        accuracy[test_name] = acc

        print(f"    evaluated {len(is_right)} rows, accuracy={acc:.4f}  |  wrongs saved → {write_path}")

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

    # Use the standard (non-shuffled) answer key for aggregation indexing
    answer_path = os.path.join(args.answer_dir, "finalized_ucsf_pdgm_pairs.csv")
    answer_df = pd.read_csv(answer_path)
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
# STAGE 3 — Average accuracy across multi-run experiments per model type
# ──────────────────────────────────────────────────────────────────────────────
def _parse_run_info(test_name: str) -> tuple[str, str, int | None]:
    """
    Parse a test_name like 'MedGemma1.5_multi_slice_run2' into
    (model_name, base_test, run_number).  If no _runN suffix, returns None.

    Heuristic: the model dir name is the first path component, and
    _runN is stripped from the stem.  E.g.:
        'MedGemma1.5_multi_slice_results_run2'
      → model_dir='MedGemma1.5', base='multi_slice_results', run=2
    """
    import re as _re
    run_match = _re.search(r'_run(\d+)$', test_name)
    if run_match:
        run_num = int(run_match.group(1))
        base = test_name[:run_match.start()]
    else:
        run_num = None
        base = test_name
    return base, run_num


def stage3_average_runs(args: argparse.Namespace) -> dict[str, dict]:
    """
    Average accuracy across multi-run results for each model type.

    Reads Evals.json (from Stage 1), groups entries by model type and
    base test name (stripping _runN), and produces:
      - Per-test averaged accuracy
      - Per-model-type averaged accuracy (mean across all tests)
    """
    print("\n" + "=" * 60)
    print("STAGE 3 — Averaging accuracy across runs per model type")
    print("=" * 60)

    evals_path = os.path.join(args.metrics_dir, "Evals.json")
    if not os.path.exists(evals_path):
        print("  [warn] Evals.json not found — run Stage 1 first.")
        return {}

    with open(evals_path, "r") as f:
        accuracy = json.load(f)

    # Group by base test name (stripping _runN)
    from collections import defaultdict
    test_runs: dict[str, list[float]] = defaultdict(list)
    model_tests: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for test_name, acc in accuracy.items():
        base, run_num = _parse_run_info(test_name)
        test_runs[base].append(acc)

        # Try to extract model name from the directory structure
        # test_name format is typically "ModelDir_testname" from iter_result_csvs
        # The model dir is the parent folder name baked into test_name
        parts = test_name.split("_")
        # Walk the result CSVs to find the model directory
        # For simplicity, use the first component before the base test name
        # e.g., 'MedGemma1.5_multi_slice_run1' → model='MedGemma1.5'
        for tname, read_path, _ in iter_result_csvs(args.qa_path):
            if tname == test_name:
                model_dir = read_path.split('/')[-2]  # parent directory name
                base_test = base.replace(f"{model_dir}_", "", 1)
                model_tests[model_dir][base_test].append(acc)
                break

    # Compute averages
    averaged_results: dict[str, dict] = {}

    for model_name, tests in model_tests.items():
        test_averages = {}
        all_test_means = []

        for test_base, accs in tests.items():
            mean_acc = sum(accs) / len(accs)
            test_averages[test_base] = {
                "runs": len(accs),
                "individual_accuracies": [round(a, 4) for a in accs],
                "mean_accuracy": round(mean_acc, 4),
            }
            all_test_means.append(mean_acc)
            print(f"  {model_name}/{test_base}: {len(accs)} runs, mean={mean_acc:.4f}")

        model_mean = sum(all_test_means) / len(all_test_means) if all_test_means else 0.0
        averaged_results[model_name] = {
            "model_mean_accuracy": round(model_mean, 4),
            "num_tests": len(tests),
            "tests": test_averages,
        }
        print(f"  → {model_name} overall mean: {model_mean:.4f} ({len(tests)} tests)")

    # Also compute simple per-test averages (without model grouping) as fallback
    simple_averages = {}
    for base, accs in test_runs.items():
        simple_averages[base] = {
            "runs": len(accs),
            "mean_accuracy": round(sum(accs) / len(accs), 4),
        }

    # Write results
    os.makedirs(args.metrics_dir, exist_ok=True)

    averaged_path = os.path.join(args.metrics_dir, "Averaged_Evals.json")
    output = {
        "per_model": averaged_results,
        "per_test": simple_averages,
    }
    with open(averaged_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"\n  ✓ Averaged results saved → {averaged_path}")

    return averaged_results

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
        stage3_average_runs(args)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()