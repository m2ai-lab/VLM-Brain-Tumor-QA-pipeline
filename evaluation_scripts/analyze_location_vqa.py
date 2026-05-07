import pandas as pd
import glob
import os
import re
import sys
from collections import Counter
import math
try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ── Dynamic Config Resolution ──────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

def calculate_stats(correct, total, p0=0.25):
    """
    Calculates Z-score and p-value for a proportion test.
    p0: null hypothesis proportion (default 0.25 for 4-choice questions)
    """
    if total == 0:
        return 0.0, 1.0
    
    p_hat = correct / total
    
    # Standard Error for null hypothesis
    se = math.sqrt((p0 * (1 - p0)) / total)
    
    if se == 0:
        return 0.0, 1.0
    
    z = (p_hat - p0) / se
    
    if SCIPY_AVAILABLE:
        # Two-tailed p-value
        p_val = 2 * (1 - norm.cdf(abs(z)))
    else:
        # Simple approximation if scipy is missing
        # z=1.96 approx p=0.05
        p_val = -1.0 
        
    return z, p_val

def analyze_category(pattern, base_path):
    """
    Analyzes a specific category (e.g., 'text_only' or 'blank') across all models.
    """
    results = {}
    correct_questions = {}
    all_questions = {}
    num_runs = {}
    
    qa_path = _cfg.get("qa_path")
    qa_df = pd.read_csv(qa_path) if qa_path and os.path.exists(qa_path) else None

    search_pattern = os.path.join(base_path, f"**/*{pattern}*.csv")
    all_files = glob.glob(search_pattern, recursive=True)
    target_files = [f for f in all_files if "wrongs" not in f]

    for file_path in target_files:
        # Determine model name from directory or filename if needed
        model_name = os.path.dirname(file_path).split(os.sep)[-1]
        if model_name == "Responses" or model_name == "Results" or not model_name:
            # Fallback to filename prefix if directory is generic
            fname = os.path.basename(file_path)
            if "_" in fname:
                model_name = fname.split("_")[0]
            else:
                model_name = "Root"
            
        if model_name not in results:
            results[model_name] = {'correct': 0, 'total': 0}
            correct_questions[model_name] = Counter()
            all_questions[model_name] = set()
            num_runs[model_name] = 0
            
        num_runs[model_name] += 1
            
        try:
            df = pd.read_csv(file_path)
            
            # Handle missing Question/Answer columns
            if qa_df is not None and len(df) == len(qa_df):
                if "Question" not in df.columns:
                    df["Question"] = qa_df["Question"]
                if "Answer" not in df.columns:
                    df["Answer"] = qa_df["Answer"]
            
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
            
    return results, correct_questions, all_questions, num_runs

def run_analysis():
    base_path = _cfg.get("output_base")
    if not base_path:
        print("Error: output_base not found in config.")
        return

    # Chance level (null hypothesis)
    p0 = 0.25 # Assuming 4 choices on average

    print(f"Analyzing files... (Null Hypothesis Chance Level: {p0})")
    text_results, text_correct_maps, text_qs, text_runs = analyze_category("text_only", base_path)
    blank_results, blank_correct_maps, blank_qs, blank_runs = analyze_category("blank", base_path)

    all_models = sorted(set(text_results.keys()) | set(blank_results.keys()))

    # Enhanced Header for Statistics
    header = f"{'Model':<18} | {'Text: Acc':<12} | {'T-Z':<6} | {'T-p':<6} | {'Blank: Acc':<12} | {'B-Z':<6} | {'B-p':<6}"
    print("\n" + "="*len(header))
    print(header)
    print("-" * len(header))

    model_blank_correct = {}

    for model in all_models:
        t_res = text_results.get(model, {'correct': 0, 'total': 0})
        b_res = blank_results.get(model, {'correct': 0, 'total': 0})
        
        t_z, t_p = calculate_stats(t_res['correct'], t_res['total'], p0)
        b_z, b_p = calculate_stats(b_res['correct'], b_res['total'], p0)
        
        t_acc = f"{t_res['correct']}/{t_res['total']}"
        b_acc = f"{b_res['correct']}/{b_res['total']}"
        
        # Formatting p-values
        t_p_str = f"{t_p:.3f}" if t_p >= 0.001 else "<.001"
        b_p_str = f"{b_p:.3f}" if b_p >= 0.001 else "<.001"
        if t_p < 0: t_p_str = "N/A"
        if b_p < 0: b_p_str = "N/A"

        print(f"{model:<18} | {t_acc:<12} | {t_z:>6.2f} | {t_p_str:<6} | {b_acc:<12} | {b_z:>6.2f} | {b_p_str:<6}")
        
        model_blank_correct[model] = set(blank_correct_maps.get(model, {}).keys())

    print("="*len(header))
    if not SCIPY_AVAILABLE:
        print("\nNOTE: scipy not found. p-values are not calculated. Please install scipy for full stats.")

    # Questions EVERY model got right in BLANK mode
    print("\n" + "="*85)
    print("QUESTIONS EVERY MODEL GOT RIGHT (Blank Mode Only)")
    print("="*85)
    
    if all_models:
        first_model = all_models[0]
        always_correct_blank = model_blank_correct.get(first_model, set()).copy()
        for model in all_models[1:]:
            always_correct_blank = always_correct_blank.intersection(model_blank_correct.get(model, set()))
            
        if not always_correct_blank:
            print("No questions were answered correctly by every model in Blank mode.")
        else:
            print(f"Total Questions Always Correct in Blank: {len(always_correct_blank)}")
            for i, q in enumerate(sorted(list(always_correct_blank))):
                q_display = (q[:120] + '...') if len(q) > 120 else q
                print(f"  {i+1}. {q_display}")

    # Top 3 Overlapping
    print("\n" + "="*85)
    print("TOP 3 OVERLAPPING QUESTIONS PER MODEL (Correct in both Text and Blank)")
    print("="*85)
    
    for model in all_models:
        text_q_map = text_correct_maps.get(model, {})
        blank_q_map = blank_correct_maps.get(model, {})
        overlap_qs = set(text_q_map.keys()).intersection(set(blank_q_map.keys()))
        if not overlap_qs: continue
        print(f"\n>>> MODEL: {model}")
        sorted_overlap = sorted(overlap_qs, key=lambda q: text_q_map[q] + blank_q_map[q], reverse=True)
        for i, q in enumerate(sorted_overlap[:8]):
            combined_hits = text_q_map[q] + blank_q_map[q]
            q_display = (q[:100] + '...') if len(q) > 200 else q
            print(f"  {i+1}. [Hits: {combined_hits}] {q_display}")

    # SPECIFIC: Qwen Text-Only Overlap Across All Runs
    print("\n" + "="*85)
    print("QUESTIONS QWEN GOT RIGHT ON ALL TEXT-ONLY RUNS")
    print("="*85)
    
    qwen_models = [m for m in all_models if "qwen" in m.lower()]
    for model in qwen_models:
        runs = text_runs.get(model, 0)
        if runs <= 1: continue # Only interesting if multiple runs
        
        correct_map = text_correct_maps.get(model, {})
        always_correct = [q for q, count in correct_map.items() if count == runs]
        
        print(f"\n>>> MODEL: {model} ({runs} runs analyzed)")
        if not always_correct:
            print("  No 'location' questions were answered correctly across all runs.")
        else:
            print(f"  Total consistent 'location' answers: {len(always_correct)}")
            for i, q in enumerate(sorted(always_correct)[:15]): # Show top 15
                q_display = (q[:120] + '...') if len(q) > 120 else q
                print(f"    {i+1}. {q_display}")

    print("\n" + "="*85)

if __name__ == "__main__":
    run_analysis()
