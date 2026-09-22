# Archived analysis implementation; paths and metric conventions are preserved.
# See tools/evaluate_saved.py for evaluation with an explicit input and output.
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. ADAPTIVE BALANCED CCC
# ==========================================
def ccc_score(y_true, y_pred):
    """Compute Lin's concordance correlation coefficient."""
    if np.std(y_true) <= 1e-6 or np.std(y_pred) <= 1e-6:
        return np.nan
        
    cor = np.corrcoef(y_true, y_pred)[0][1]
    if np.isnan(cor):
        return np.nan
        
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    sd_true, sd_pred = np.std(y_true), np.std(y_pred)
    
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    if denominator == 0:
        return np.nan
        
    numerator = 2 * cor * sd_true * sd_pred
    return numerator / denominator

def bootstrap_adaptive_balanced_ccc(gt, pred, bins, n_iterations=1000, min_power=10):
    """Estimate balanced CCC by resampling equally from occupied score intervals.
Use median observed bin support with the supplied minimum sample count."""
    valid_bins = [b for b in bins if not (b[0] == 0.0 and b[1] == bins[-1][1])]
    indices_por_bin = []
    tamanos_reales = []
    
    for min_g, max_g in valid_bins:
        if min_g == valid_bins[0][0]:
            mask = (gt >= min_g) & (gt <= max_g)
        else:
            mask = (gt > min_g) & (gt <= max_g)
            
        idx = np.where(mask)[0]
        if len(idx) > 0:
            indices_por_bin.append(idx)
            tamanos_reales.append(len(idx))
            
    if len(indices_por_bin) < 2:
        return np.nan, np.nan

    # ==========================================================
    # ADAPTIVE PER-INTERVAL SAMPLE SIZE BASED ON MEDIAN SUPPORT
    # ==========================================================
    # Compute the median of the observed class counts.
    n_target_per_interval = int(np.median(tamanos_reales))
    
    # Set a lower bound on the bootstrap sample count.
    # If the median is small, use min_power as the lower bound.
    n_target_per_interval = max(min_power, n_target_per_interval)
    # ==========================================================

    ccc_scores = []
    
    for _ in range(n_iterations):
        muestra_gt = []
        muestra_pred = []
        
        for idx_array in indices_por_bin:
            # Sample the adaptive number of observations with replacement.
            # This upsamples or downsamples each bin relative to its observed support.
            idx_sampleados = np.random.choice(idx_array, size=n_target_per_interval, replace=True)
            muestra_gt.extend(gt[idx_sampleados])
            muestra_pred.extend(pred[idx_sampleados])
            
        muestra_gt = np.array(muestra_gt)
        muestra_pred = np.array(muestra_pred)
        
        if np.std(muestra_gt) > 1e-6 and np.std(muestra_pred) > 1e-6:
            ccc_val = ccc_score(muestra_gt, muestra_pred)
            if not np.isnan(ccc_val):
                ccc_scores.append(ccc_val)
                
    if not ccc_scores:
        return np.nan, n_target_per_interval
        
    return np.mean(ccc_scores), n_target_per_interval

# ==========================================
# 2. EXPERIMENT CONFIGURATION
# ==========================================
bins_10 = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0), (0.0, 10.0)]
bins_5_10 = [(0.0, 7.5), (7.5, 10.0), (0.0, 10.0)]

num_runs = 5
MULTIPLICADOR = 5.0
ITERACIONES_BOOTSTRAP = 1000
MINIMA_POTENCIA = 20 # Minimum resampled observations per interval; does not add independent evidence.

resultados_totales = []

# ==========================================
# 3. LOAD ARRAYS AND CALCULATE METRICS
# ==========================================
print(f"[INFO] Iniciando evaluación rigurosa (CCC con Balanceo Adaptativo por Mediana, {ITERACIONES_BOOTSTRAP} iteraciones)...")

for run in range(1, num_runs + 1):
    dir_media = f"../05_hmamba/exp/hmamba-CV-DB/0/fold_{run}/preds"

    
    try:
        gt_phn = np.load(f"{dir_media}/phn_target.npy").flatten()
        gt_word = np.load(f"{dir_media}/word_target.npy")
        gt_utt = np.load(f"{dir_media}/utt_target.npy")

        pred_phn = np.load(f"{dir_media}/phn_pred.npy").squeeze().flatten()
        pred_word = np.load(f"{dir_media}/word_pred.npy") 
        pred_utt = np.load(f"{dir_media}/utt_pred.npy")
        
        aspectos_config = [
            ("FONEMA", gt_phn, pred_phn, bins_10),
            ("W-ACC", gt_word[:, 0], pred_word[:, 0], bins_10),
            ("W-STRESS", gt_word[:, 1], pred_word[:, 1], bins_5_10),
            ("W-TOTAL", gt_word[:, 2], pred_word[:, 2], bins_10),
            ("U-ACC", gt_utt[:, 0], pred_utt[:, 0], bins_10),
            ("U-COMP", gt_utt[:, 1], pred_utt[:, 1], bins_5_10),
            ("U-FLU", gt_utt[:, 2], pred_utt[:, 2], bins_10),
            ("U-PRO", gt_utt[:, 3], pred_utt[:, 3], bins_10),
            ("U-TOT", gt_utt[:, 4], pred_utt[:, 4], bins_10)
        ]
        
        for nombre, gt_raw, pred_raw, bins in aspectos_config:
            mask_valid = gt_raw >= 0
            gt_clean = gt_raw[mask_valid] * MULTIPLICADOR
            pred_clean = pred_raw[mask_valid] * MULTIPLICADOR
            
            # The function returns both the metric and selected sample count.
            ccc_m, n_target_aspect = bootstrap_adaptive_balanced_ccc(
                gt_clean, pred_clean, bins, n_iterations=ITERACIONES_BOOTSTRAP, min_power=MINIMA_POTENCIA
            )
            
            resultados_totales.append({
                'Run': run,
                'Aspect': nombre,
                'CCC_Bal': ccc_m,
                'N_Target': n_target_aspect # Record the sample count used.
            })
            
    except FileNotFoundError:
        print(f"[ERROR] Archivos no encontrados para el Run {run}. Abortando iteración.")
        continue

# ==========================================
# 4. AGGREGATE RUNS AND REPORT RESULTS
# ==========================================
df_results = pd.DataFrame(resultados_totales)

# Group by aspect; average PCC/CCC and retain the per-aspect sample count.
df_stats = df_results.groupby('Aspect').agg({
    'CCC_Bal': ['mean', 'std'],
    'N_Target': 'first' # Take the first value, assuming the same target support across runs.
}).reset_index()

df_stats.columns = ['Aspect', 'CCC_mean', 'CCC_std', 'N_Target']

orden_aspectos = ["FONEMA", "W-ACC", "W-STRESS", "W-TOTAL", "U-ACC", "U-COMP", "U-FLU", "U-PRO", "U-TOT"]
df_stats['Aspect'] = pd.Categorical(df_stats['Aspect'], categories=orden_aspectos, ordered=True)
df_stats = df_stats.sort_values('Aspect')

print("\n" + "="*80)
print(f" RENDIMIENTO (CCC BALANCEADO CON N ADAPTATIVO POR MEDIANA)")
print("="*80)

col_w_m = 28
col_w_n = 25
header = f"{'ASPECTO':<12} | {'CCC (MEDIA ± DESV)':^{col_w_m}} | {'N_OBJETIVO (POR INTERVALO)':^{col_w_n}}"
print(header)
print("-" * len(header))

for _, row in df_stats.iterrows():
    asp = row['Aspect']
    ccc_m, ccc_s = row['CCC_mean'], row['CCC_std']
    n_target = row['N_Target']
    
    row_str = f"{asp:<12} | "
    
    if pd.isna(ccc_m):
        row_str += f"{'INVIABLE (Datos Vacíos)':^{col_w_m}} | "
        if pd.isna(n_target):
            row_str += f"{'---':^{col_w_n}}"
        else:
            row_str += f"{n_target:^{col_w_n}.0f}"
    else:
        row_str += f"{ccc_m:.3f} ± {ccc_s:.3f}".center(col_w_m) + " | "
        row_str += f"{n_target:^{col_w_n}.0f}"
        
    print(row_str)

print("="*80 + "\n")