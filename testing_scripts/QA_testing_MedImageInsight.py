"""
QA_testing_MedImageInsight.py — Zero-shot VQA using MedImageInsight.

MedImageInsight is a CLIP-style contrastive embedding model, NOT a generative
VLM. It cannot "answer" questions — instead, it computes cosine similarity
between an image embedding and text embeddings for candidate labels.

Strategy:
  1. Parse each question to extract the numbered answer options.
  2. Encode the patient's MRI slice with the vision encoder.
  3. Encode each answer option (optionally prefixed with the question for
     context) with the text encoder.
  4. Pick the option with the highest similarity score as the predicted answer.

This gives us a zero-shot classification baseline on the brain MRI QA task
using only image–text alignment, with no reasoning capability.
"""
import os
import sys
import re
import base64
import argparse
import pandas as pd

# ── Ensure the MedImageInsights repo is importable ────────────────────────────
# The model code lives inside the downloaded repo and needs its own subpackages.


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
    from PIL import Image
    import io

    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.encodebytes(buf.getvalue()).decode("utf-8")


def query_the_model(classifier, question: str, patient_id: str, base_image_dir: str):
    """
    Perform zero-shot classification on a single patient image using the
    question's answer options as candidate labels.
    """
    # 1. Locate the image
    patient_image_path = os.path.join(base_image_dir, str(patient_id), "Axial.png")

    if not os.path.exists(patient_image_path):
        return {
            "reasoning": f"Error: Image {patient_image_path} not found.",
            "answer": "Error",
        }

    # 2. Parse the 4 answer options from the question
    try:
        question_stem, options = parse_answer_options(question)
    except ValueError as e:
        return {
            "reasoning": f"Parse error: {e}",
            "answer": "Error",
        }

    # 3. Encode the image
    image_b64 = read_image_as_base64(patient_image_path)

    # 4. Use the question stem + each option as labels for richer context
    #    e.g., "Based on the T2/FLAIR hyperintensity, 1) Low Grade"
    #    This gives the text encoder more semantic signal than bare labels.
    labels = [f"{question_stem} {opt}" for opt in options]

    # 5. Zero-shot classification
    results = classifier.predict([image_b64], labels)
    # results is a list of dicts: [{label: probability, ...}]

    if not results:
        return {"reasoning": "Model returned empty results.", "answer": "Error"}

    probs = results[0]  # dict: {label: prob}

    # 6. Find the best match and map back to the original option
    best_label = max(probs, key=probs.get)
    best_idx = labels.index(best_label)
    predicted_option = options[best_idx]

    # Build reasoning string from all probabilities
    prob_lines = []
    for label, opt in zip(labels, options):
        prob_lines.append(f"  {opt}: {probs.get(label, 0.0):.4f}")
    reasoning = "Zero-shot cosine similarity scores:\n" + "\n".join(prob_lines)

    return {"reasoning": reasoning, "answer": predicted_option}


def query_the_model_blank(classifier, question: str, patient_id: str, image_path: str):
    """Same as query_the_model but for the blank (control) experiment."""
    if not os.path.exists(image_path):
        return {
            "reasoning": f"Error: Image {image_path} not found.",
            "answer": "Error",
        }

    try:
        question_stem, options = parse_answer_options(question)
    except ValueError as e:
        return {"reasoning": f"Parse error: {e}", "answer": "Error"}

    image_b64 = read_image_as_base64(image_path)
    labels = [f"{question_stem} {opt}" for opt in options]

    results = classifier.predict([image_b64], labels)
    if not results:
        return {"reasoning": "Empty results.", "answer": "Error"}

    probs = results[0]
    best_label = max(probs, key=probs.get)
    best_idx = labels.index(best_label)
    predicted_option = options[best_idx]

    prob_lines = [f"  {opt}: {probs.get(label, 0.0):.4f}" for label, opt in zip(labels, options)]
    reasoning = "Zero-shot cosine similarity scores:\n" + "\n".join(prob_lines)

    return {"reasoning": reasoning, "answer": predicted_option}


def main(args):
    # ── Add MedImageInsights repo to sys.path so its internal imports work ──
    sys.path.insert(0, args.model_path)

    from medimageinsightmodel import MedImageInsight

    # model_dir must be absolute — the model code uses it in os.path.join
    # for config.yaml, vision weights, and tokenizer paths.
    model_dir = os.path.join(args.model_path, "2024.09.27")

    print(f"Loading MedImageInsight from {model_dir}...")
    classifier = MedImageInsight(
        model_dir=model_dir,
        vision_model_name="medimageinsigt-v1.0.0.pt",
        language_model_name="language_model.pth",
    )
    classifier.load_model()

    qa_data = pd.read_csv(args.qa_path)
    generated_answer = []
    generated_reasoning = []
    total = qa_data.shape[0]

    for idx, row in qa_data.iterrows():
        print(f"Processing {idx+1}/{total} (ID: {row['Assigned ID']})...")

        if args.image_path:
            # Blank variant — use a fixed image for all questions
            response = query_the_model_blank(
                classifier, row["Question"], row["Assigned ID"], args.image_path
            )
        else:
            # Normal variant — per-patient image directory
            response = query_the_model(
                classifier, row["Question"], row["Assigned ID"], args.image_dir
            )

        generated_answer.append(response["answer"])
        generated_reasoning.append(response["reasoning"])
        print(f"Response: {response['answer']}\n{'-'*30}")

    # Save results
    qa_data["predicted_answer"] = generated_answer
    qa_data["MedImageInsight_Reasoning"] = generated_reasoning

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    qa_data.to_csv(args.output_path, index=False)
    print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedImageInsight Zero-Shot VQA")
    parser.add_argument('--qa_path', type=str,
                        default="/scratch/group/CX000019_DS1/vlm-brain-mri/finalized_ucsf_pdgm_pairs.csv")
    parser.add_argument('--output_path', type=str,
                        default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/MedImageInsight/single_slice_results.csv")
    parser.add_argument('--image_dir', type=str,
                        default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/format_dataset/2D_slices")
    parser.add_argument('--image_path', type=str, default=None,
                        help="For blank experiments: path to a single blacked-out image.")
    parser.add_argument('--model_path', type=str,
                        default="/mnt/scratch/group/CX000019_DS1/vlm-brain-mri/MedImageInsights")

    args = parser.parse_args()
    main(args)
