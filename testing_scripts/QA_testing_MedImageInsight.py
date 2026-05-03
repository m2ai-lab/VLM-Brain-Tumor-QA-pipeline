"""
QA_testing_MedImageInsight.py — Zero-shot VQA using MedImageInsight with batch processing.

MedImageInsight is a CLIP-style contrastive embedding model, NOT a generative
VLM. It cannot "answer" questions — instead, it computes cosine similarity
between an image embedding and text embeddings for candidate labels.

Strategy:
  1. Parse each question to extract the numbered answer options.
  2. Encode the patient's MRI slice with the vision encoder.
  3. Encode each answer option (optionally prefixed with the question for
     context) with the text encoder.
  4. Pick the option with the highest similarity score as the predicted answer.

Batching:
  Because each question has different label text, we call classifier.predict()
  with a batch of N images and their per-image label lists simultaneously.
  This amortizes the vision encoder cost across the whole batch.
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config
_cfg = load_config()
import re
import base64
import argparse
import pandas as pd
from io import BytesIO
from PIL import Image

from testing_scripts.utils.checkpoint import load_checkpoint, save_checkpoint, get_row_id


# ── Ensure the MedImageInsights repo is importable ────────────────────────────


def parse_answer_options(question: str) -> tuple[str, list[str]]:
    """
    Split a 4-option multiple-choice question into its stem and options.

    Uses the same proven regex as dataset_reshuffle.py.

    Returns:
        (question_stem, ["1) Option A", "2) Option B", "3) Option C", "4) Option D"])
    """
    parts = re.split(r"([0-5]\))", question)
    # parts = [stem, "1)", "Option A ", "2)", "Option B ", ...]
    stem = parts[0].strip()

    # Re-join the number markers with their text
    options = []
    for i in range(1, len(parts) - 1, 2):
        marker = parts[i]           # e.g., "1)"
        text = parts[i + 1].strip() # e.g., "Option A"
        options.append(f"{marker} {text}")

    if len(options) != 4:
        raise ValueError(
            f"Expected 4 answer options, got {len(options)} from: {question[:120]}..."
        )
    return stem, options


def read_image_as_base64(image_path: str) -> str:
    """Read an image file, force RGB, and return its base64-encoded string."""
    img = Image.open(image_path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.encodebytes(buf.getvalue()).decode("utf-8")


def run_batch(
    classifier,
    batch_rows: list[dict],
    base_image_dir: str,
    image_filename: str,
    image_path_override: str | None = None,
) -> list[dict]:
    """
    Zero-shot classification for a batch of QA rows.

    Collects all (image_b64, labels) pairs then calls classifier.predict()
    with all N images at once.  Rows with missing images or parse errors
    are handled individually without blocking the rest of the batch.
    """
    valid_indices: list[int] = []
    results: list[dict] = [{"reasoning": "Not processed", "answer": "Error"}] * len(batch_rows)

    images_b64: list[str] = []
    per_image_labels: list[list[str]] = []
    per_image_options: list[list[str]] = []

    for i, row in enumerate(batch_rows):
        if image_path_override:
            patient_image_path = image_path_override
        else:
            patient_image_path = os.path.join(base_image_dir, str(row["Assigned ID"]), image_filename)

        if not os.path.exists(patient_image_path):
            results[i] = {
                "reasoning": f"Error: Image {patient_image_path} not found.",
                "answer": "Error",
            }
            continue

        try:
            question_stem, options = parse_answer_options(row["Question"])
        except ValueError as e:
            results[i] = {"reasoning": f"Parse error: {e}", "answer": "Error"}
            continue

        image_b64 = read_image_as_base64(patient_image_path)
        labels = [f"{question_stem} {opt}" for opt in options]

        valid_indices.append(i)
        images_b64.append(image_b64)
        per_image_labels.append(labels)
        per_image_options.append(options)

    if not valid_indices:
        return results

    # Batch predict: classifier receives N images each with their own label list.
    # MedImageInsight.predict() signature: predict(images: list[str], labels: list[str])
    # where labels is a flat list applied to all images uniformly.
    # Since our labels differ per image, we call predict once per image but still
    # batch the image encoding by calling predict with all images and a union label set,
    # then remapping.  If the API only supports uniform labels, fall back to per-image calls.
    try:
        # Attempt batch call with uniform label union (richer context)
        all_labels_flat = list({lbl for lbls in per_image_labels for lbl in lbls})
        batch_results = classifier.predict(images_b64, all_labels_flat)
        use_batch = True
    except Exception:
        use_batch = False

    for seq_idx, row_idx in enumerate(valid_indices):
        options = per_image_options[seq_idx]
        labels = per_image_labels[seq_idx]

        if use_batch and batch_results:
            probs = batch_results[seq_idx]  # dict: {label: prob}
            # Only consider labels belonging to this image's question
            relevant_probs = {lbl: probs.get(lbl, 0.0) for lbl in labels}
        else:
            # Fallback: individual predict call
            single_result = classifier.predict([images_b64[seq_idx]], labels)
            if not single_result:
                results[row_idx] = {"reasoning": "Model returned empty results.", "answer": "Error"}
                continue
            relevant_probs = single_result[0]

        if not relevant_probs:
            results[row_idx] = {"reasoning": "Model returned empty results.", "answer": "Error"}
            continue

        best_label = max(relevant_probs, key=relevant_probs.get)
        # Map best_label back to the short option text
        try:
            best_idx = labels.index(best_label)
            predicted_option = options[best_idx]
        except ValueError:
            # best_label came from union; match by suffix
            predicted_option = next(
                (opt for lbl, opt in zip(labels, options) if lbl == best_label),
                "Error"
            )

        prob_lines = [f"  {opt}: {relevant_probs.get(lbl, 0.0):.4f}"
                      for lbl, opt in zip(labels, options)]
        reasoning = "Zero-shot cosine similarity scores:\n" + "\n".join(prob_lines)

        results[row_idx] = {"reasoning": reasoning, "answer": predicted_option}

    return results


def main(args):
    # ── Add MedImageInsights repo to sys.path so its internal imports work ──
    sys.path.insert(0, args.model_path)

    from medimageinsightmodel import MedImageInsight

    # model_dir must be absolute — the model code uses it in os.path.join
    model_dir = os.path.join(args.model_path, "2024.09.27")

    print(f"Loading MedImageInsight from {model_dir}...")
    classifier = MedImageInsight(
        model_dir=model_dir,
        vision_model_name="medimageinsigt-v1.0.0.pt",
        language_model_name="language_model.pth",
    )
    classifier.load_model()

    qa_data = pd.read_csv(args.qa_path)
    if getattr(args, 'limit', None):
        print(f"Limiting to first {args.limit} rows.")
        qa_data = qa_data.head(args.limit)

    completed_ids = load_checkpoint(args.output_path)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} rows already completed, skipping.")

    total = len(qa_data)
    batch_size = args.batch_size
    print(f"Running MedImageInsight inference with batch_size={batch_size} on {total} rows.")

    rows_list = qa_data.to_dict("records")
    df_index = list(range(len(rows_list)))

    for batch_start in range(0, total, batch_size):
        batch_records = rows_list[batch_start: batch_start + batch_size]
        batch_df_indices = df_index[batch_start: batch_start + batch_size]

        pending = [
            (i, rec) for i, rec in zip(batch_df_indices, batch_records)
            if get_row_id(rec["Assigned ID"], rec["Question"]) not in completed_ids
        ]
        if not pending:
            continue

        pending_df_indices, pending_records = zip(*pending)
        batch_end = min(batch_start + batch_size, total)
        print(f"Processing rows {batch_start + 1}–{batch_end}/{total} "
              f"(batch of {len(pending_records)})...")

        responses = run_batch(
            classifier,
            list(pending_records),
            base_image_dir=args.image_dir,
            image_filename=args.image_filename,
            image_path_override=args.image_path,
        )

        batch_df = qa_data.iloc[list(pending_df_indices)].copy()
        save_checkpoint(
            args.output_path,
            batch_df,
            {
                "predicted_answer":         [r["answer"]    for r in responses],
                "MedImageInsight_Reasoning": [r["reasoning"] for r in responses],
            },
        )
        completed_ids.update(get_row_id(rec["Assigned ID"], rec["Question"]) for rec in pending_records)
        print(f"  → Checkpoint saved ({len(completed_ids)}/{total} total done).")

    print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedImageInsight Zero-Shot VQA")
    parser.add_argument('--qa_path', type=str,
                        default=_cfg.get("qa_path"))
    parser.add_argument('--output_path', type=str,
                        default=_cfg.get("output_base", "") + "/MedImageInsight/single_slice_results.csv")
    parser.add_argument('--image_dir', type=str,
                        default=_cfg.get("slice_dir"))
    parser.add_argument('--image_filename', type=str, default="Axial.png",
                        help="Filename inside each patient dir. Use 'axial_slices_montage.png' for montage.")
    parser.add_argument('--image_path', type=str, default=None,
                        help="For blank experiments: path to a single blacked-out image.")
    parser.add_argument('--model_path', type=str,
                        default=_cfg.get("medimageinsight_model_path", "models/MedImageInsights"))
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Number of images per classifier.predict() call. No GPU generation so can be large.")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of rows for testing")

    args = parser.parse_args()
    main(args)
