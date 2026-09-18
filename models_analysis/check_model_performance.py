# Archived analysis implementation; paths and metric conventions are preserved.
# See tools/evaluate_saved.py for evaluation with an explicit input and output.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. INTERVAL CONFIGURATION & STRUCTURES
# ==========================================
bins_10 = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0), (0.0, 10.0)]
bins_5_10 = [(0.0, 7.5), (7.5, 10.0), (0.0, 10.0)]

# Labels explicitly formatted with correct mathematical topology
labels_5_10 = ['[4.0, 6.0)', '[8.0, 10.0]', 'Total']

train_support_data = []
test_error_data = []

# ==========================================
# 2. CORE EVALUATION FUNCTIONS
# ==========================================
def compute_train_support(aspect_name, gt_train_array, multiplier, bins, labels=None):
    """
    Calculates support (N) on the training set to populate the distribution heatmap.
    Uses strict [a, b) boundary logic to perfectly match np.histogram outputs.
    """
    mask_valid = gt_train_array >= 0
    gt_clean = gt_train_array[mask_valid]
    gt_scaled = gt_clean * multiplier

    for idx, (min_g, max_g) in enumerate(bins):
        # Strict [a, b) topology emulation
        if min_g == 0.0 and max_g == 10.0:
            mask = (gt_scaled >= min_g) & (gt_scaled <= max_g)
        elif max_g == 10.0:
            mask = (gt_scaled >= min_g) & (gt_scaled <= max_g)
        else:
            mask = (gt_scaled >= min_g) & (gt_scaled < max_g)
        
        n = np.sum(mask)
        is_total = (min_g == 0.0 and max_g == bins[-1][1])
        
        # Dynamic label generation applying formal interval notation
        if labels and idx < len(labels):
            range_str = labels[idx]
        else:
            if min_g == 0.0 and max_g == 10.0:
                range_str = f"[{min_g}, {max_g}]"
            elif max_g == 10.0:
                range_str = f"[{min_g}, {max_g}]"
            else:
                range_str = f"[{min_g}, {max_g})"
        
        train_support_data.append({
            'Aspect': aspect_name.upper(),
            'Range': range_str,
            'N_Train': n,
            'Is_Total': is_total
        })

def evaluate_test_mse(run_idx, aspect_name, gt_test_array, pred_test_array, gt_multiplier, pred_multiplier, bins, labels=None):
    """
    Calculates Mean Squared Error (MSE) strictly on the test set inference tensors.
    Applies symmetric scaling to evaluate errors in the true 0-10 score space.
    """
    mask_valid = gt_test_array >= 0
    gt_clean = gt_test_array[mask_valid]
    pred_clean = pred_test_array[mask_valid]
    
    gt_scaled = gt_clean * gt_multiplier
    pred_scaled = pred_clean * pred_multiplier

    for idx, (min_g, max_g) in enumerate(bins):
        # Strict [a, b) topology emulation
        if min_g == 0.0 and max_g == 10.0:
            mask = (gt_scaled >= min_g) & (gt_scaled <= max_g)
        elif max_g == 10.0:
            mask = (gt_scaled >= min_g) & (gt_scaled <= max_g)
        else:
            mask = (gt_scaled >= min_g) & (gt_scaled < max_g)
        
        gt_seg = gt_scaled[mask]
        pred_seg = pred_scaled[mask]
        
        if len(gt_seg) < 1:
            mse = np.nan
        else:
            diff = pred_seg - gt_seg
            mse = np.mean(diff**2)
            
        is_total = (min_g == 0.0 and max_g == bins[-1][1])
        
        # Dynamic label generation applying formal interval notation
        if labels and idx < len(labels):
            range_str = labels[idx]
        else:
            if min_g == 0.0 and max_g == 10.0:
                range_str = f"[{min_g}, {max_g}]"
            elif max_g == 10.0:
                range_str = f"[{min_g}, {max_g}]"
            else:
                range_str = f"[{min_g}, {max_g})"
        
        test_error_data.append({
            'Run': run_idx,
            'Aspect': aspect_name.upper(),
            'Range': range_str,
            'MSE': mse,
            'Is_Total': is_total
        })

# ==========================================
# 3. PHASE A: TRAIN SET SUPPORT (N)
# ==========================================
print("[INFO] Phase A: Extracting support densities from the Training set...")
dir_gt = "../06_M3C/data/GoP_VC_Features_Embbeding_norm"

gt_phn_raw = np.load(f"{dir_gt}/tr_label_phn.npy")[:, :, 1]
valid_mask = gt_phn_raw != -1
gt_phn_train = gt_phn_raw[valid_mask].flatten()

gt_word_raw = np.load(f"{dir_gt}/tr_label_word.npy")
word_id = gt_word_raw[:, :, -1]
target = gt_word_raw[:, :, 0:3]

valid_token_target = []
for i in range(target.shape[0]):
    prev_w_id = 0
    start_id = 0
    for j in range(target.shape[1]):
        cur_w_id = int(word_id[i, j])
        if cur_w_id != prev_w_id:
            valid_token_target.append(np.mean(target[i, start_id:j, :], axis=0))
            if cur_w_id == -1: 
                break
            else: 
                prev_w_id = cur_w_id
                start_id = j
gt_word_train = np.array(valid_token_target).round(2)

gt_utt_train = np.load(f"{dir_gt}/tr_label_utt.npy")

compute_train_support("PHONEME", gt_phn_train, multiplier=5.0, bins=bins_10)
compute_train_support("W-ACC", gt_word_train[:, 0], multiplier=1.0, bins=bins_10)
compute_train_support("W-STR", gt_word_train[:, 1], multiplier=1.0, bins=bins_5_10, labels=labels_5_10)
compute_train_support("W-TOT", gt_word_train[:, 2], multiplier=1.0, bins=bins_10)
compute_train_support("U-ACC", gt_utt_train[:, 0], multiplier=1.0, bins=bins_10)
compute_train_support("U-COM", gt_utt_train[:, 1], multiplier=1.0, bins=bins_5_10, labels=labels_5_10)
compute_train_support("U-FLU", gt_utt_train[:, 2], multiplier=1.0, bins=bins_10)
compute_train_support("U-PRO", gt_utt_train[:, 3], multiplier=1.0, bins=bins_10)
compute_train_support("U-TOT", gt_utt_train[:, 4], multiplier=1.0, bins=bins_10)

# ==========================================
# 4. PHASE B: TEST SET ERROR (MSE)
# ==========================================
print("[INFO] Phase B: Evaluating Test MSE across 5 execution runs...")
num_runs = 5

for run in range(1, num_runs + 1):
    print(f"[INFO] Evaluating run {run}...")
    mean_dir = f"../04_ConPCO/exp/conpco-BOMA/fold_{run}/preds"
    
    try:
        gt_phn_test = np.load(f"{mean_dir}/phn_target.npy").flatten()
        gt_word_test = np.load(f"{mean_dir}/word_target.npy")
        gt_utt_test = np.load(f"{mean_dir}/utt_target.npy")

        pred_phn_test = np.load(f"{mean_dir}/phn_pred.npy").squeeze().flatten()
        pred_word_test = np.load(f"{mean_dir}/word_pred.npy")
        pred_utt_test = np.load(f"{mean_dir}/utt_pred.npy")
    except FileNotFoundError:
        print(f"FATAL ERROR: Missing test files in run {run} directory. Execution aborted.")
        exit()

    evaluate_test_mse(run, "PHONEME", gt_phn_test, pred_phn_test, 5.0, 5.0, bins_10)
    
    evaluate_test_mse(run, "W-ACC", gt_word_test[:, 0], pred_word_test[:, 0], 5.0, 5.0, bins_10)
    evaluate_test_mse(run, "W-STR", gt_word_test[:, 1], pred_word_test[:, 1], 5.0, 5.0, bins_5_10, labels=labels_5_10)
    evaluate_test_mse(run, "W-TOT", gt_word_test[:, 2], pred_word_test[:, 2], 5.0, 5.0, bins_10)
    
    evaluate_test_mse(run, "U-ACC", gt_utt_test[:, 0], pred_utt_test[:, 0], 5.0, 5.0, bins_10)
    evaluate_test_mse(run, "U-COM", gt_utt_test[:, 1], pred_utt_test[:, 1], 5.0, 5.0, bins_5_10, labels=labels_5_10)
    evaluate_test_mse(run, "U-FLU", gt_utt_test[:, 2], pred_utt_test[:, 2], 5.0, 5.0, bins_10)
    evaluate_test_mse(run, "U-PRO", gt_utt_test[:, 3], pred_utt_test[:, 3], 5.0, 5.0, bins_10)
    evaluate_test_mse(run, "U-TOT", gt_utt_test[:, 4], pred_utt_test[:, 4], 5.0, 5.0, bins_10)

# ==========================================
# 5. PHASE C: STATISTICAL CORRELATION & FUSION
# ==========================================
print("[INFO] Phase C: Fusing data and calculating topological distance correlation...")
df_support = pd.DataFrame(train_support_data)
df_error = pd.DataFrame(test_error_data)

df_support_bins = df_support[~df_support['Is_Total']].copy()
df_error_bins = df_error[~df_error['Is_Total']].copy()

df_error_stats = df_error_bins.groupby(['Aspect', 'Range']).agg({
    'MSE': ['mean', 'std']
}).reset_index()
df_error_stats.columns = ['Aspect', 'Range', 'MSE_mean', 'MSE_std']

df_final = pd.merge(df_support_bins, df_error_stats, on=['Aspect', 'Range'], how='inner')

# Explicitly ordered columns utilizing the accurate mathematical notation
ordered_cols = ["[0.0, 2.0)", "[2.0, 4.0)", "[4.0, 6.0)", "[6.0, 8.0)", "[8.0, 10.0]"]
range_to_idx = {val: idx for idx, val in enumerate(ordered_cols)}
df_final['Range_Idx'] = df_final['Range'].map(range_to_idx)

majority_indices = df_final.loc[df_final.groupby('Aspect')['N_Train'].idxmax()][['Aspect', 'Range_Idx']]
majority_indices = majority_indices.rename(columns={'Range_Idx': 'Majority_Idx'})

df_final = df_final.merge(majority_indices, on='Aspect', how='left')
df_final['Distance_D'] = (df_final['Range_Idx'] - df_final['Majority_Idx']).abs()

corr_list = []
for aspect, sub_df in df_final.groupby('Aspect'):
    sub_clean = sub_df.dropna(subset=['Distance_D', 'MSE_mean'])
    
    if len(sub_clean) > 2 and sub_clean['Distance_D'].std() > 0:
        corr, p_val = stats.spearmanr(sub_clean['Distance_D'], sub_clean['MSE_mean'])
    else:
        corr, p_val = np.nan, np.nan
        
    corr_list.append({
        'Aspect': aspect, 
        'Spearman_Rho': corr,
        'p_value': p_val,
        'Majority_Bin': ordered_cols[int(sub_clean['Majority_Idx'].iloc[0])] if len(sub_clean) > 0 else 'N/A'
    })

df_corr_stats = pd.DataFrame(corr_list)

global_clean = df_final.dropna(subset=['Distance_D', 'MSE_mean'])
global_spearman, global_p = stats.spearmanr(global_clean['Distance_D'], global_clean['MSE_mean'])

# ==========================================
# 6. PHASE D: VISUALIZATION (HEATMAPS)
# ==========================================
ordered_aspects = sorted(df_final['Aspect'].unique())

n_train_p = df_final.pivot(index='Aspect', columns='Range', values='N_Train').reindex(index=ordered_aspects, columns=ordered_cols).fillna(0)
mse_mean_p = df_final.pivot(index='Aspect', columns='Range', values='MSE_mean').reindex(index=ordered_aspects, columns=ordered_cols)
mse_std_p = df_final.pivot(index='Aspect', columns='Range', values='MSE_std').reindex(index=ordered_aspects, columns=ordered_cols)

annot_n = n_train_p.copy().astype(str)
annot_mse = mse_mean_p.copy().astype(str)

for r in ordered_aspects:
    for c in ordered_cols:
        val_n = n_train_p.loc[r, c]
        m_m, s_m = mse_mean_p.loc[r, c], mse_std_p.loc[r, c]
        
        annot_n.loc[r, c] = "0" if val_n == 0 else f"{val_n:.0f}"
        annot_mse.loc[r, c] = "---" if pd.isna(m_m) else f"{m_m:.2f}\n±{s_m:.2f}"

sns.set_theme(style="white")
fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1.15]})

axes[0].set_facecolor('#f2f2f2')
axes[1].set_facecolor('#f2f2f2')

sns.heatmap(n_train_p, annot=annot_n, fmt="", cmap="Greys", 
            norm=SymLogNorm(linthresh=1, vmin=0, base=10),
            cbar_kws={'label': 'Train Support N (SymLog Scale)'}, ax=axes[0], linewidths=.5, cbar=True)
axes[0].set_title("Training Set Distribution (N)", fontweight='bold', pad=15)
axes[0].set_xlabel("Score Range", fontweight='bold')
axes[0].set_ylabel("Acoustic Aspect", fontweight='bold')

sns.heatmap(mse_mean_p, annot=annot_mse, fmt="", cmap="Reds", vmin=0, vmax=60,
            cbar_kws={'label': 'Test Mean Squared Error (MSE)', 
                      'ticks': [0, 10, 20, 30, 40, 50]}, 
            ax=axes[1], linewidths=.5)
axes[1].set_title("Prediction Error on Test Set (Mean MSE ± Std)", fontweight='bold', pad=15)
axes[1].set_xlabel("Score Range", fontweight='bold')
axes[1].set_ylabel("") 

plt.tight_layout()
plt.savefig("mse_vs_n_analysis_split.png", dpi=300, bbox_inches='tight')
print("[INFO] Comparative chart 'mse_vs_n_analysis_split.png' successfully exported.")

# ==========================================
# 7. CORRELATION SUMMARY OUTPUT
# ==========================================
print("\n" + "="*95)
print(" SUMMARY: SPEARMAN CORRELATION BETWEEN DISTANCE TO MAJORITY BIN (D) AND ERROR (MSE)")
print("="*95)

header = f"{'ASPECT':<15} | {'SPEARMAN RHO':<15} | {'P-VALUE':<15}"
print(header)
print("-" * len(header))

for aspect in ordered_aspects:
    row = df_corr_stats[df_corr_stats['Aspect'] == aspect]
    if not row.empty:
        maj_bin = row['Majority_Bin'].values[0]
        rho = row['Spearman_Rho'].values[0]
        pval = row['p_value'].values[0]
        
        if pd.isna(rho):
             print(f"{aspect:<15} | {'---':<15} | {'---':<15}")
        else:
             print(f"{aspect:<15} | {rho:<15.3f} | {pval:<15.4e}")

print("-" * len(header))
print(f"GLOBAL SPEARMAN RHO (All Aspects combined): {global_spearman:.3f} (p-value: {global_p:.4e})")
print("=" * 95)