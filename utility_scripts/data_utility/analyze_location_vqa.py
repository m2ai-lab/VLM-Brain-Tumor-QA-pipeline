import pandas as pd
import glob
import os
import re
import sys
from collections import Counter

# ── Dynamic Config Resolution ──────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:   
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config
_cfg = load_config()

def normalize(text):
    """
    Normalizes answer strings to handle cases like '2) Answer' vs 'Answer'.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    # Remove common prefixes like "1) ", "a. ", "1. ", "1- "
    text = re.sub(r'^[a-z0-9][\s\).\-]+', '', text)
    return text.strip()

def analyze_category(pattern, base_path):
    """
    Analyzes a specific category (e.g., 'text_only' or 'blank') across all models.
    Returns: 
      - results: {model: {correct, total}}
      - correct_questions: {model: Counter({question_text: count_correct})}
      - all_questions: {model: set(all_location_questions)}
    """
    results = {}
    correct_questions = {}
    all_questions = {}
    
    qa_path = _cfg.get("qa_path")
    qa_df = pd.read_csv(qa_path) if qa_path and os.path.exists(qa_path) else None

    search_pattern = os.path.join(base_path, f"**/*{pattern}*.csv")
    all_files = glob.glob(search_pattern, recursive=True)
    target_files = [f for f in all_files if "wrongs" not in f]

    for file_path in target_files:
        model_name = os.path.dirname(file_path).split(os.sep)[-1]
        if not model_name:
            model_name = "Root"
            
        if model_name not in results:
            results[model_name] = {'correct': 0, 'total': 0}
            correct_questions[model_name] = Counter()
            all_questions[model_name] = set()
            
        try:
            df = pd.read_csv(file_path)
            
            # Handle missing Question column (added by user)
            if "Question" not in df.columns and qa_df is not None:
                if len(df) == len(qa_df):
                    df["Question"] = qa_df["Question"]
            
            required = ['Question', 'Answer', 'predicted_answer']
            if not all(col in df.columns for col in required):
                continue
                
            mask = df['Question'].str.contains(r'location|where', case=False, na=False)
            subset = df[mask]
            
            results[model_name]['total'] += len(subset)
            
            for _, row in subset.iterrows():
                q_text = row['Question'].strip()
                all_questions[model_name].add(q_text)
                
                if normalize(row['Answer']) == normalize(row['predicted_answer']):
                    results[model_name]['correct'] += 1
                    correct_questions[model_name][q_text] += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    return results, correct_questions, all_questions

def run_analysis():
    base_path = _cfg.get("output_base")
    if not base_path:
        print("Error: output_base not found in config.")
        return

    print("Analyzing files... please wait.")
    text_results, text_correct_maps, _ = analyze_category("text_only", base_path)
    blank_results, blank_correct_maps, blank_all_maps = analyze_category("blank", base_path)

    all_models = sorted(set(text_results.keys()) | set(blank_results.keys()))

    print("\n" + "="*85)
    print(f"{'Model':<20} | {'Text Only':<12} | {'Blank':<12} | {'Overlap Count'}")
    print("-" * 85)

    for model in all_models:
        text_stats = text_results.get(model, {'correct': 0, 'total': 0})
        blank_stats = blank_results.get(model, {'correct': 0, 'total': 0})
        
        text_q_set = set(text_correct_maps.get(model, {}).keys())
        blank_q_set = set(blank_correct_maps.get(model, {}).keys())
        overlap_qs = text_q_set.intersection(blank_q_set)
        
        text_str = f"{text_stats['correct']}/{text_stats['total']}"
        blank_str = f"{blank_stats['correct']}/{blank_stats['total']}"
        
        print(f"{model:<20} | {text_str:<12} | {blank_str:<12} | {len(overlap_qs)}")

    print("="*85)

    # NEW: Questions every model got wrong in Blank mode
    print("\n" + "="*85)
    print("QUESTIONS EVERY MODEL GOT WRONG (Blank Mode)")
    print("="*85)
    
    # 1. Identify all unique location questions seen in any blank run
    all_blank_qs = set()
    for q_set in blank_all_maps.values():
        all_blank_qs.update(q_set)
        
    # 2. Identify all questions that were EVER answered correctly in blank mode by ANY model
    ever_correct_blank = set()
    for q_map in blank_correct_maps.values():
        ever_correct_blank.update(q_map.keys())
        
    # 3. Intersection of (All Blank Questions) and (Not Ever Correct)
    always_wrong_blank = all_blank_qs - ever_correct_blank
    
    if not always_wrong_blank:
        print("No questions were wrong for every model across all blank runs!")
    else:
        print(f"Total Questions Always Wrong: {len(always_wrong_blank)}")
        for i, q in enumerate(sorted(list(always_wrong_blank))[:10]): # Show top 10 for brevity
            q_display = (q[:100] + '...') if len(q) > 100 else q
            print(f"  {i+1}. {q_display}")
        if len(always_wrong_blank) > 10:
            print(f"  ... and {len(always_wrong_blank) - 10} more.")

    # Display Top 3 Overlapping Questions per Model
    print("\n" + "="*85)
    print("TOP 3 OVERLAPPING QUESTIONS (Correct in both Text-Only and Blank)")
    print("="*85)
    
    for model in all_models:
        text_q_map = text_correct_maps.get(model, {})
        blank_q_map = blank_correct_maps.get(model, {})
        overlap_qs = set(text_q_map.keys()).intersection(set(blank_q_map.keys()))
        
        if not overlap_qs:
            continue
            
        print(f"\n>>> MODEL: {model}")
        sorted_overlap = sorted(overlap_qs, key=lambda q: text_q_map[q] + blank_q_map[q], reverse=True)
        for i, q in enumerate(sorted_overlap[:3]):
            combined_hits = text_q_map[q] + blank_q_map[q]
            q_display = (q[:100] + '...') if len(q) > 100 else q
            print(f"  {i+1}. [Hits: {combined_hits}] {q_display}")

    print("\n" + "="*85)

if __name__ == "__main__":
    run_analysis()
