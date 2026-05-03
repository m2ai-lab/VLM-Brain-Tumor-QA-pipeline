"""
testing_scripts/utils/checkpoint.py — Checkpoint/resume utility for VLM inference scripts.

All testing scripts use the same CSV output format.  If a job is preempted or
killed mid-run, this module lets you resume from the last completed batch rather
than restarting from scratch.

Usage pattern in a testing script
----------------------------------
    from testing_scripts.utils.checkpoint import load_checkpoint, save_checkpoint

    completed_ids = load_checkpoint(args.output_path)

    for batch in batches:
        # skip rows whose ID+Question was already saved
        batch = [r for r in batch if get_row_id(r["Assigned ID"], r["Question"]) not in completed_ids]
        if not batch:
            continue
        results = run_model(batch)
        save_checkpoint(args.output_path, batch_df_slice, results)
        completed_ids.update(get_row_id(r["Assigned ID"], r["Question"]) for r in batch)
"""
from __future__ import annotations

import os
import pandas as pd
from typing import Any


def get_row_id(assigned_id: Any, question: str) -> str:
    """Combine Assigned ID and Question into a unique identifier."""
    return f"{assigned_id}|||{question}"


def load_checkpoint(output_path: str) -> set:
    """
    Return the set of get_row_id(ID, Question) values already written to output_path.

    If the file does not exist yet (fresh run), returns an empty set.
    The output CSV is written by save_checkpoint() in append mode, so this
    correctly picks up any partial progress from a previous run.
    """
    if not os.path.exists(output_path):
        return set()

    try:
        df = pd.read_csv(output_path)
        if "Assigned ID" in df.columns and "Question" in df.columns:
            return set(
                df.apply(lambda r: get_row_id(r["Assigned ID"], r["Question"]), axis=1).tolist()
            )
    except Exception as e:
        print(f"[checkpoint] Warning: could not read checkpoint at {output_path}: {e}")

    return set()


def save_checkpoint(
    output_path: str,
    rows: pd.DataFrame,
    extra_columns: dict[str, list[Any]],
) -> None:
    """
    Atomically append completed rows to the output CSV.

    Parameters
    ----------
    output_path   : Final output CSV path (also used as the checkpoint file).
    rows          : The slice of qa_data DataFrame for this batch (same index order).
    extra_columns : Dict mapping new column names to lists of values, e.g.
                    {"predicted_answer": [...], "MedGemma_Reasoning": [...]}.
                    Must have the same length as len(rows).
    """
    chunk = rows.copy()
    for col, values in extra_columns.items():
        chunk[col] = values

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    write_header = not os.path.exists(output_path)
    chunk.to_csv(output_path, mode="a", header=write_header, index=False)
