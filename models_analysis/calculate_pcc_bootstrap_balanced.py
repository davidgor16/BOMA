# Archived analysis implementation; paths and metric conventions are preserved.
# See tools/evaluate_saved.py for evaluation with an explicit input and output.
import numpy as np
import warnings
import os
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# ==========================================
# 1. ADAPTIVE BOOTSTRAP PCC
# ==========================================
def bootstrap_adaptive_balanced_pcc(gt, pred, bins, n_iterations=1000, min_power=10):
    """Estimate balanced PCC by adaptive resampling from mutually exclusive score intervals."""
    indices_por_bin = []
    tamanos_reales = []
    
    for min_g, max_g in bins:
        if min_g == bins[0][0]:
            mask = (gt >= min_g) & (gt <= max_g)
        else:
            mask = (gt > min_g) & (gt <= max_g)
            
        idx = np.where(mask)[0]
        if len(idx) > 0:
            indices_por_bin.append(idx)
            tamanos_reales.append(len(idx))
            
    if len(indices_por_bin) < 2:
        return np.nan, np.nan

    # Compute the per-bin sample count from median support.
    n_target_per_interval = int(np.median(tamanos_reales))
    n_target_per_interval = max(min_power, n_target_per_interval)

    pcc_scores = []
    
    for _ in range(n_iterations):
        muestra_gt = []
        muestra_pred = []
        
        for idx_array in indices_por_bin:
            idx_sampleados = np.random.choice(idx_array, size=n_target_per_interval, replace=True)
            muestra_gt.extend(gt[idx_sampleados])
            muestra_pred.extend(pred[idx_sampleados])
            
        muestra_gt = np.array(muestra_gt)
        muestra_pred = np.array(muestra_pred)
        
        if np.std(muestra_gt) > 1e-6 and np.std(muestra_pred) > 1e-6:
            pcc, _ = pearsonr(muestra_gt, muestra_pred)
            if not np.isnan(pcc):
                pcc_scores.append(pcc)
                
    if not pcc_scores:
        return np.nan, n_target_per_interval
        
    return np.mean(pcc_scores), n_target_per_interval


# ==========================================
# 2. EXPERIMENT CONFIGURATION
# ==========================================
base_dir = "../../exp/HMAMBA"
num_runs = 5
dir_template = base_dir + "/{run}/preds" 
ITERACIONES_BOOTSTRAP = 1000
MINIMA_POTENCIA = 15

# Use mutually exclusive intervals.
bins_10 = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0)]
bins_2 = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]

aspects_config = {
    'phn':      {'name': 'Fonema - Puntuación',      'scale': 1.0, 'max_c': 2,  'level': 'phn',  'idx': None, 'bins': bins_2},
    'word_acc': {'name': 'Palabra - Accuracy',       'scale': 5.0, 'max_c': 10, 'level': 'word', 'idx': 0,    'bins': bins_10},
    'word_str': {'name': 'Palabra - Stress',         'scale': 5.0, 'max_c': 10, 'level': 'word', 'idx': 1,    'bins': bins_10},
    'word_tot': {'name': 'Palabra - Total',          'scale': 5.0, 'max_c': 10, 'level': 'word', 'idx': 2,    'bins': bins_10},
    'utt_acc':  {'name': 'Frase - Accuracy',         'scale': 5.0, 'max_c': 10, 'level': 'utt',  'idx': 0,    'bins': bins_10},
    'utt_comp': {'name': 'Frase - Completeness',     'scale': 5.0, 'max_c': 10, 'level': 'utt',  'idx': 1,    'bins': bins_10},
    'utt_flu':  {'name': 'Frase - Fluency',          'scale': 5.0, 'max_c': 10, 'level': 'utt',  'idx': 2,    'bins': bins_10},
    'utt_pro':  {'name': 'Frase - Prosodic',         'scale': 5.0, 'max_c': 10, 'level': 'utt',  'idx': 3,    'bins': bins_10},
    'utt_tot':  {'name': 'Frase - Total',            'scale': 5.0, 'max_c': 10, 'level': 'utt',  'idx': 4,    'bins': bins_10}
}

metrics_history = {
    key: {'pcc_bal': [], 'mse': [], 'n_target': []} for key in aspects_config.keys()
}

# ==========================================
# 3. LOAD ARRAYS AND CALCULATE METRICS
# ==========================================
print(f"[INFO] Ejecutando evaluación rigurosa (PCC-Bootstrap Adaptativo, {ITERACIONES_BOOTSTRAP} iteraciones)...\n")

for run in range(1, num_runs + 1):
    dir_run = dir_template.format(run=run)
    
    if not os.path.exists(dir_run):
        print(f"[ERROR] Directorio inaccesible en fold {run}. Abortando extracción para esta partición: {dir_run}")
        continue
        
    gt_phn = np.load(f"{dir_run}/phn_target.npy").flatten()
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
            
        y_true_s = y_true_raw * cfg['scale']
        # Artificial clipping would hide prediction errors outside the target range.
        y_pred_s = np.clip(y_pred_raw * cfg['scale'], 0, cfg['max_c']) 
        
        valid_mask = ~np.isnan(y_pred_s)
        y_true_s = y_true_s[valid_mask]
        y_pred_s = y_pred_s[valid_mask]
        
        # 3.1 Calculate balanced bootstrap PCC.
        pcc_bal, n_targ = bootstrap_adaptive_balanced_pcc(
            y_true_s, y_pred_s, cfg['bins'], 
            n_iterations=ITERACIONES_BOOTSTRAP, 
            min_power=MINIMA_POTENCIA
        )
        
        metrics_history[key]['pcc_bal'].append(pcc_bal)
        if not pd.isna(n_targ):
            metrics_history[key]['n_target'].append(n_targ)

        # 3.2 Calculate global MSE.
        if len(y_true_s) > 0:
            mse = np.mean((y_true_s - y_pred_s)**2)
        else:
            mse = np.nan
            
        metrics_history[key]['mse'].append(mse)

# ==========================================
# 4. AGGREGATE RUNS AND REPORT RESULTS
# ==========================================
print(f"\n{'='*105}")
print(f" RENDIMIENTO ESTRUCTURAL: PCC-BOOTSTRAP vs MSE ({num_runs} RUNS) ".center(105))
print(f"{'='*105}")
print(f"{'ASPECTO':<28} | {'PCC_BAL (Media ± Desv)':<25} | {'N_OBJETIVO (Mediana)':<22} | {'MSE (Media ± Desv)':<22}")
print("-" * 105)

for key, cfg in aspects_config.items():
    hist = metrics_history[key]
    
    if not hist['pcc_bal'] or not hist['mse']:
        print(f" {cfg['name']:<27} | {'FALLO (Sin Datos)':<25} | {'---':<22} | {'---':<22}")
        continue
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pcc_arr = np.array(hist['pcc_bal'], dtype=float)
        mse_arr = np.array(hist['mse'], dtype=float)
        n_arr = np.array(hist['n_target'], dtype=float)
        
        pcc_m, pcc_s = np.nanmean(pcc_arr), np.nanstd(pcc_arr)
        mse_m, mse_s = np.nanmean(mse_arr), np.nanstd(mse_arr)
        n_m = np.nanmedian(n_arr) if len(n_arr) > 0 else np.nan
        
    if pd.isna(pcc_m):
        pcc_str = f"{'INVIABLE':<25}"
        n_str = f"{'---':<22}"
    else:
        pcc_str = f"{pcc_m:.3f} ± {pcc_s:.3f}"
        n_str = f"{n_m:.0f} / intervalo"

    if pd.isna(mse_m):
        mse_str = f"{'INVIABLE':<22}"
    else:
        mse_str = f"{mse_m:.3f} ± {mse_s:.3f}"
        
    print(f" {cfg['name']:<27} | {pcc_str:<25} | {n_str:<22} | {mse_str:<22}")

print("=" * 105)