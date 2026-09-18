# Archived analysis implementation; paths and metric conventions are preserved.
# See tools/evaluate_saved.py for evaluation with an explicit input and output.
import numpy as np
import warnings
import os
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# ==========================================
# 1. EXPERIMENT CONFIGURATION (5 RUNS)
# ==========================================
base_dir = "../06_M3C/exp/M3C-BOMA"
num_runs = 5
dir_template = base_dir + "/fold_{run}/preds" 

aspects_config = {
    'phn':      {'name': 'Fonema - Puntuación',      'scale': 1.0, 'max_c': 2,  'level': 'phn',  'idx': None},
    'word_acc': {'name': 'Palabra - Accuracy',       'scale': 5.0, 'max_c': 10, 'level': 'word', 'idx': 0},
    'word_str': {'name': 'Palabra - Stress',         'scale': 5.0, 'max_c': 10, 'level': 'word', 'idx': 1},
    'word_tot': {'name': 'Palabra - Total',          'scale': 5.0, 'max_c': 10, 'level': 'word', 'idx': 2},
    'utt_acc':  {'name': 'Frase - Accuracy',         'scale': 5.0, 'max_c': 10, 'level': 'utt',  'idx': 0},
    'utt_comp': {'name': 'Frase - Completeness',     'scale': 5.0, 'max_c': 10, 'level': 'utt',  'idx': 1},
    'utt_flu':  {'name': 'Frase - Fluency',          'scale': 5.0, 'max_c': 10, 'level': 'utt',  'idx': 2},
    'utt_pro':  {'name': 'Frase - Prosodic',         'scale': 5.0, 'max_c': 10, 'level': 'utt',  'idx': 3},
    'utt_tot':  {'name': 'Frase - Total',            'scale': 5.0, 'max_c': 10, 'level': 'utt',  'idx': 4}
}


# Initialize the accumulators for all metrics.
metrics_history = {
    key: {'pcc': [], 'mse': []} for key in aspects_config.keys()
}

# ==========================================
# 2. LOAD ARRAYS AND CALCULATE METRICS
# ==========================================
print(f"Iniciando cálculo iterativo para {num_runs} runs...\n")

for run in range(1, num_runs + 1):
    dir_run = dir_template.format(run=run)
    
    if not os.path.exists(dir_run):
        print(f"⚠️ ADVERTENCIA: Ausencia de directorio en iteración {run}. Ruta: {dir_run}")
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
        y_pred_s = np.clip(y_pred_raw * cfg['scale'], 0, cfg['max_c']) 
        
        valid_mask = ~np.isnan(y_pred_s)
        y_true_s = y_true_s[valid_mask]
        y_pred_s = y_pred_s[valid_mask]
        
        # Calculate PCC.
        if len(y_true_s) > 1:
            if np.std(y_true_s) == 0 or np.std(y_pred_s) == 0:
                pcc = 0.0
            else:
                pcc, _ = pearsonr(y_true_s, y_pred_s)
        else:
            pcc = np.nan
        
        metrics_history[key]['pcc'].append(pcc)

        # Calculate unstratified MSE.
        if len(y_true_s) > 0:
            mse = np.mean((y_true_s - y_pred_s)**2)
        else:
            mse = np.nan
            
        metrics_history[key]['mse'].append(mse)

# ==========================================
# 3. GLOBAL SUMMARY TABLE
# ==========================================
print(f"\n{'='*85}")
print(f" RESUMEN GLOBAL DE MÉTRICAS ({num_runs} RUNS) ".center(85))
print(f"{'='*85}")
print(f"{'Aspecto Evaluado':<30} | {'PCC (Media ± Std) ↑':<25} | {'MSE (Media ± Std) ↓':<25}")
print("-" * 85)

for key, cfg in aspects_config.items():
    hist = metrics_history[key]
    if not hist['pcc'] or not hist['mse']:
        continue
        
    pcc_arr = np.array(hist['pcc'])
    mse_arr = np.array(hist['mse'])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pcc_m, pcc_s = np.nanmean(pcc_arr), np.nanstd(pcc_arr)
        mse_m, mse_s = np.nanmean(mse_arr), np.nanstd(mse_arr)
        
    pcc_str = f"{pcc_m:.3f} ± {pcc_s:.3f}"
    mse_str = f"{mse_m:.3f} ± {mse_s:.3f}"
    
    print(f" {cfg['name']:<29} | {pcc_str:<25} | {mse_str:<25}")

print("=" * 85)
print("\n--- EJECUCIÓN COMPLETADA ---")