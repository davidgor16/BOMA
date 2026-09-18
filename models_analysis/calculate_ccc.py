# Archived analysis implementation; paths and metric conventions are preserved.
# See tools/evaluate_saved.py for evaluation with an explicit input and output.
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. STANDARD CCC
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

# ==========================================
# 2. EXPERIMENT CONFIGURATION
# ==========================================
num_runs = 5
MULTIPLICADOR = 5.0
resultados_totales = []

# ==========================================
# 3. LOAD ARRAYS AND CALCULATE METRICS
# ==========================================
print("[INFO] Iniciando evaluación rigurosa (CCC global directo, sin remuestreo)...")

for run in range(num_runs):
    dir_media = f"../../exp/HMAMBA/{run}/preds"
    
    try:
        gt_phn = np.load(f"{dir_media}/phn_target.npy").flatten()
        gt_word = np.load(f"{dir_media}/word_target.npy")
        gt_utt = np.load(f"{dir_media}/utt_target.npy")

        pred_phn = np.load(f"{dir_media}/phn_pred.npy").squeeze().flatten()
        pred_word = np.load(f"{dir_media}/word_pred.npy") 
        pred_utt = np.load(f"{dir_media}/utt_pred.npy")
        
        aspectos_config = [
            ("FONEMA", gt_phn, pred_phn),
            ("W-ACC", gt_word[:, 0], pred_word[:, 0]),
            ("W-STRESS", gt_word[:, 1], pred_word[:, 1]),
            ("W-TOTAL", gt_word[:, 2], pred_word[:, 2]),
            ("U-ACC", gt_utt[:, 0], pred_utt[:, 0]),
            ("U-COMP", gt_utt[:, 1], pred_utt[:, 1]),
            ("U-FLU", gt_utt[:, 2], pred_utt[:, 2]),
            ("U-PRO", gt_utt[:, 3], pred_utt[:, 3]),
            ("U-TOT", gt_utt[:, 4], pred_utt[:, 4])
        ]
        
        for nombre, gt_raw, pred_raw in aspectos_config:
            # Apply the validity mask and score multiplier.
            mask_valid = gt_raw >= 0
            gt_clean = gt_raw[mask_valid] * MULTIPLICADOR
            pred_clean = pred_raw[mask_valid] * MULTIPLICADOR
            
            # Calculate directly over the complete valid array.
            ccc_val = ccc_score(gt_clean, pred_clean)
            
            resultados_totales.append({
                'Run': run,
                'Aspect': nombre,
                'CCC': ccc_val
            })
            
    except FileNotFoundError:
        print(f"[ERROR] Archivos no encontrados para el Run {run}. Abortando iteración.")
        continue

# ==========================================
# 4. AGGREGATE RUNS AND REPORT RESULTS
# ==========================================
df_results = pd.DataFrame(resultados_totales)

if df_results.empty:
    print("[ERROR] Ausencia de datos viables para la agregación estadística.")
else:
    df_stats = df_results.groupby('Aspect').agg({
        'CCC': ['mean', 'std']
    }).reset_index()

    df_stats.columns = ['Aspect', 'CCC_mean', 'CCC_std']

    orden_aspectos = ["FONEMA", "W-ACC", "W-STRESS", "W-TOTAL", "U-ACC", "U-COMP", "U-FLU", "U-PRO", "U-TOT"]
    df_stats['Aspect'] = pd.Categorical(df_stats['Aspect'], categories=orden_aspectos, ordered=True)
    df_stats = df_stats.sort_values('Aspect')

    print("\n" + "="*50)
    print(" RENDIMIENTO (CCC GLOBAL NORMAL)")
    print("="*50)

    col_w_m = 28
    header = f"{'ASPECTO':<12} | {'CCC (MEDIA ± DESV)':^{col_w_m}}"
    print(header)
    print("-" * len(header))

    for _, row in df_stats.iterrows():
        asp = row['Aspect']
        ccc_m, ccc_s = row['CCC_mean'], row['CCC_std']
        
        row_str = f"{asp:<12} | "
        
        if pd.isna(ccc_m):
            row_str += f"{'INVIABLE (Datos Vacíos)':^{col_w_m}}"
        else:
            row_str += f"{ccc_m:.3f} ± {ccc_s:.3f}".center(col_w_m)
            
        print(row_str)

    print("="*50 + "\n")