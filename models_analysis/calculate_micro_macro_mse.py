# Archived analysis implementation; paths and metric conventions are preserved.
# See tools/evaluate_saved.py for evaluation with an explicit input and output.
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

# ==========================================
# 1. MACRO-MSE FUNCTION (STRICT IMPLEMENTATION)
# ==========================================
def calc_macro_mse(preds, targets, edges, min_support=5):
    """
    Calculates the Macro-MSE (Mean Squared Error stratified by ranges).
    Enforces a minimum support threshold to prevent high-variance estimates
    from outlier bins.
    """
    indices = np.digitize(targets, edges)
    num_bins = len(edges) + 1
    
    mse_per_range = []
    
    for k in range(num_bins):
        mask = (indices == k)
        # Strict enforcement of statistical minimum support
        if np.sum(mask) >= min_support:
            mse_k = np.mean((preds[mask] - targets[mask]) ** 2)
            mse_per_range.append(mse_k)
            
    # Returns np.nan instead of 0.0 to avoid artificial zero-error distortion
    if not mse_per_range:
        return np.nan 
    return float(np.mean(mse_per_range))

# ==========================================
# 2. EXPERIMENT CONFIGURATION (5 RUNS)
# ==========================================
base_dir = "../05_hmamba/exp/hmamba-BOMA-SB/0"
num_runs = 5
dir_template = base_dir + "/fold_{run}/preds" 

# Phn strictly scaled to 0-10 (scale: 5.0). Edges unified to match word/utt accuracy.
aspects_config = {
    'phn':      {'name': 'Phoneme - Score',        'edges': [2.0, 4.0, 6.0, 8.0], 'labels': ['[0, 2)', '[2, 4)', '[4, 6)', '[6, 8)', '[8, 10]'], 'scale': 5.0, 'max_c': 10, 'level': 'phn',  'idx': None},
    'word_acc': {'name': 'Word - Accuracy',        'edges': [2.0, 4.0, 6.0, 8.0], 'labels': ['[0, 2)', '[2, 4)', '[4, 6)', '[6, 8)', '[8, 10]'], 'scale': 5.0, 'max_c': 10, 'level': 'word', 'idx': 0},
    'word_str': {'name': 'Word - Stress',          'edges': [7.5],                'labels': ['~ 5', '~ 10'],                                 'scale': 5.0, 'max_c': 10, 'level': 'word', 'idx': 1},
    'word_tot': {'name': 'Word - Total',           'edges': [2.0, 4.0, 6.0, 8.0], 'labels': ['[0, 2)', '[2, 4)', '[4, 6)', '[6, 8)', '[8, 10]'], 'scale': 5.0, 'max_c': 10, 'level': 'word', 'idx': 2},
    'utt_acc':  {'name': 'Utterance - Accuracy',   'edges': [2.0, 4.0, 6.0, 8.0], 'labels': ['[0, 2)', '[2, 4)', '[4, 6)', '[6, 8)', '[8, 10]'], 'scale': 5.0, 'max_c': 10, 'level': 'utt',  'idx': 0},
    'utt_comp': {'name': 'Utterance - Complete',   'edges': [7.5],                'labels': ['~ 5', '~ 10'],                                 'scale': 5.0, 'max_c': 10, 'level': 'utt',  'idx': 1},
    'utt_flu':  {'name': 'Utterance - Fluency',    'edges': [2.0, 4.0, 6.0, 8.0], 'labels': ['[0, 2)', '[2, 4)', '[4, 6)', '[6, 8)', '[8, 10]'], 'scale': 5.0, 'max_c': 10, 'level': 'utt',  'idx': 2},
    'utt_pro':  {'name': 'Utterance - Prosody',    'edges': [2.0, 4.0, 6.0, 8.0], 'labels': ['[0, 2)', '[2, 4)', '[4, 6)', '[6, 8)', '[8, 10]'], 'scale': 5.0, 'max_c': 10, 'level': 'utt',  'idx': 3},
    'utt_tot':  {'name': 'Utterance - Total',      'edges': [2.0, 4.0, 6.0, 8.0], 'labels': ['[0, 2)', '[2, 4)', '[4, 6)', '[6, 8)', '[8, 10]'], 'scale': 5.0, 'max_c': 10, 'level': 'utt',  'idx': 4}
}

metrics_history = {
    key: {
        'support': [], 
        'mse_per_class': [], 
        'macro_mse': [],
        'micro_mse': []
    } for key in aspects_config.keys()
}

# ==========================================
# 3. EXTRACTION AND ITERATIVE CALCULATION
# ==========================================
print(f"Starting Continuous Error (MSE) calculation for {num_runs} runs...\n")

# If working with k-fold cross-validation, ensure changing num_runs to (1, num_runs+1) to avoid skipping the first run.
for run in range(1, num_runs + 1):
    dir_run = dir_template.format(run=run)
    
    if not os.path.exists(dir_run):
        print(f"[WARNING] Path {dir_run} not found. Skipping run {run}...")
        continue
        
    gt_phn = np.load(f"{dir_run}/phn_target.npy").flatten()
    gt_phn
    preds_phn = np.load(f"{dir_run}/phn_pred.npy").squeeze().flatten()
    gt_word = np.load(f"{dir_run}/word_target.npy")
    preds_word = np.load(f"{dir_run}/word_pred.npy")
    gt_utt = np.load(f"{dir_run}/utt_target.npy")
    preds_utt = np.load(f"{dir_run}/utt_pred.npy")

    mask_valid = gt_phn != -1
    gt_phn = gt_phn[mask_valid]
    preds_phn = preds_phn[mask_valid]

    for key, cfg in aspects_config.items():
        if cfg['level'] == 'phn':
            y_true_raw, y_pred_raw = gt_phn, preds_phn
        elif cfg['level'] == 'word':
            y_true_raw, y_pred_raw = gt_word[:, cfg['idx']], preds_word[:, cfg['idx']]
        elif cfg['level'] == 'utt':
            y_true_raw, y_pred_raw = gt_utt[:, cfg['idx']], preds_utt[:, cfg['idx']]
            
        # 1. Scale Adjustment (Uniform scale for all levels, no artificial clipping)
        y_true_s = y_true_raw * cfg['scale']
        y_pred_s = y_pred_raw * cfg['scale']
        
        # 2. Strict NaN cleaning in predictions
        valid_mask = ~np.isnan(y_pred_s)
        y_true_s = y_true_s[valid_mask]
        y_pred_s = y_pred_s[valid_mask]
        
        # -------------------------------------------------------------
        # 3. CALCULATION VIA EXTERNAL FUNCTION (MACRO-MSE) and MICRO-MSE
        # -------------------------------------------------------------
        macro_mse = calc_macro_mse(y_pred_s, y_true_s, cfg['edges'], min_support=5)
        micro_mse = np.mean((y_true_s - y_pred_s)**2)
        
        # -------------------------------------------------------------
        # 4. REPLICATION OF LOGIC FOR DETAILED REPORTING
        # -------------------------------------------------------------
        indices = np.digitize(y_true_s, cfg['edges'])
        num_bins = len(cfg['edges']) + 1
        
        support = []
        mse_class = []
        
        for k in range(num_bins):
            mask_k = (indices == k)
            sup_k = np.sum(mask_k)
            support.append(sup_k)
            
            if sup_k >= 5:
                mse_k = np.mean((y_true_s[mask_k] - y_pred_s[mask_k])**2)
                mse_class.append(mse_k)
            else:
                mse_class.append(np.nan)
        
        metrics_history[key]['support'].append(support)
        metrics_history[key]['mse_per_class'].append(mse_class)
        metrics_history[key]['macro_mse'].append(macro_mse)
        metrics_history[key]['micro_mse'].append(micro_mse)

# ==========================================
# 4. FINAL STATISTICAL REPORT (MEAN ± STD)
# ==========================================
print(f"{'='*105}")
print(f" CROSS-VALIDATION REGRESSION REPORT ({num_runs} RUNS) - STRATIFIED MSE ".center(105))
print(f"{'='*105}")

summary_results = []

for key, cfg in aspects_config.items():
    hist = metrics_history[key]
    if len(hist['macro_mse']) == 0:
        continue
        
    supp_arr = np.array(hist['support'])
    mse_arr = np.array(hist['mse_per_class'])
    macro_arr = np.array(hist['macro_mse'], dtype=float)
    micro_arr = np.array(hist['micro_mse'], dtype=float)
    
    supp_m, supp_s = supp_arr.mean(axis=0), supp_arr.std(axis=0)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mse_m = np.nanmean(mse_arr, axis=0)
        mse_s = np.nanstd(mse_arr, axis=0)
        macro_mse_m, macro_mse_s = np.nanmean(macro_arr), np.nanstd(macro_arr)
        micro_mse_m, micro_mse_s = np.nanmean(micro_arr), np.nanstd(micro_arr)
    
    summary_results.append({
        'aspect': cfg['name'],
        'micro_m': micro_mse_m, 'micro_s': micro_mse_s,
        'macro_m': macro_mse_m, 'macro_s': macro_mse_s
    })
    
    print(f"\n>>> ASPECT: {cfg['name'].upper()}")
    print("-" * 105)
    print(f"{'Interval':<18} | {'True Support (Mean ± Std)':<30} | {'MSE (Mean Squared Error)':<30}")
    print("-" * 105)
    
    for i, label in enumerate(cfg['labels']):
        if supp_m[i] > 0:
            str_supp = f"{supp_m[i]:.1f} ± {supp_s[i]:.1f}"
            str_mse = f"{mse_m[i]:.3f} ± {mse_s[i]:.3f} pts²" if not np.isnan(mse_m[i]) else "N/A (n<5)"
            print(f" Range {label:<11} | {str_supp:<30} | {str_mse:<30}")
            
    print("-" * 105)
    print(f" [Micro] GLOBAL MSE: {micro_mse_m:.3f} ± {micro_mse_s:.3f} pts²")
    print(f" [Macro] GLOBAL MSE: {macro_mse_m:.3f} ± {macro_mse_s:.3f} pts²")
    print("=" * 105)

# ==========================================
# 5. GLOBAL SUMMARY TABLE (MICRO vs MACRO)
# ==========================================
print(f"\n\n{'='*100}")
print(f" GLOBAL SUMMARY: MICRO-MSE vs MACRO-MSE (↓ LOWER IS BETTER) ".center(100))
print(f"{'='*100}")
print(f"{'Evaluated Aspect':<35} | {'Micro-MSE':<30} | {'Macro-MSE':<25}")
print("-" * 100)

for res in summary_results:
    micro_str = f"{res['micro_m']:.3f} ± {res['micro_s']:.3f}"
    macro_str = f"{res['macro_m']:.3f} ± {res['macro_s']:.3f}"
    print(f" {res['aspect']:<34} | {micro_str:<30} | {macro_str:<25}")

print("=" * 100)
print("\n--- EXECUTION COMPLETED ---")