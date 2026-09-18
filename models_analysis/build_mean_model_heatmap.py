# Archived analysis implementation; paths and metric conventions are preserved.
# See tools/evaluate_saved.py for evaluation with an explicit input and output.
# ============================================================
# MEAN MODEL HEATMAP
#
# Produces:
#
#   mse_vs_n_all_models_ORIGINAL.png
#   mse_vs_n_all_models_BOMA.png
#
# Each heatmap contains:
#
#   IZQUIERDA:
#     Training-set distribution.
#
#   DERECHA:
#     Mean MSE across the six architectures.
#
# For each architecture:
#     Average its five runs or folds first.
#
# Then:
#     Average the six architecture-level means.
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import SymLogNorm


warnings.filterwarnings("ignore")


# ============================================================
# 1. GENERAL CONFIGURATION
# ============================================================

N_RUNS = 5

# Some HMamba experiments use 52 sequence positions.
# Use only the first 50 positions.
HMAMBA_PHN_LENGTH = 50

# Use the same color scale for ORIGINAL and BOMA.
# A shared scale makes the heatmaps visually comparable.
MSE_VMAX = 60


# ============================================================
# 2. PATHS
#
# These are the archived paths used by
# the original analysis.
#
# Update paths only if the directory names differ.
# ============================================================

MODELS = {

    "GOPT": {
        "original": Path(
            "../01_gopt/exp/gopt-original"
        ),
        "boma": Path(
            "../01_gopt/exp/gopt-BOMA"
        ),
    },

    "HiPAMA": {
        "original": Path(
            "../02_HiPAMA/exp/hipama-original"
        ),
        "boma": Path(
            "../02_HiPAMA/exp/hipama-BOMA"
        ),
    },

    "HierTFR": {
        "original": Path(
            "../03_HierTFR/exp/hiertfr-original"
        ),
        "boma": Path(
            "../03_HierTFR/exp/hiertfr-BOMA"
        ),
    },

    "ConPCO": {
        "original": Path(
            "../04_ConPCO/exp/conpco-original"
        ),
        "boma": Path(
            "../04_ConPCO/exp/conpco-BOMA"
        ),
    },

    "HMamba": {
    "original": Path(
        "../05_hmamba/exp/hmamba-original"
    ),
    "boma": Path(
        "../05_hmamba/exp/hmamba-BOMA/0"
    ),

    # Number of available runs.
    "n_runs_original": 3,
    "n_runs_boma": 5,
},

    "M3C": {
        "original": Path(
            "../06_M3C/exp/M3C-original"
        ),
        "boma": Path(
            "../06_M3C/exp/M3C-BOMA"
        ),
    },
}


# ============================================================
# 3. TRAINING GROUND-TRUTH PATHS
# ============================================================

TRAIN_GT_DIR = Path(
    "../06_M3C/data/"
    "GoP_VC_Features_Embbeding_norm"
)


# ============================================================
# 4. BINS
#
# Retain the definition used in the original script.
# ============================================================

bins_10 = [

    (0.0, 2.0),

    (2.0, 4.0),

    (4.0, 6.0),

    (6.0, 8.0),

    (8.0, 10.0),

    # Total
    (0.0, 10.0)
]


bins_5_10 = [

    (0.0, 7.5),

    (7.5, 10.0),

    # Total
    (0.0, 10.0)
]


labels_5_10 = [

    "[4.0, 6.0)",

    "[8.0, 10.0]",

    "Total"
]


# ============================================================
# 5. HEATMAP COLUMNS
# ============================================================

ORDERED_COLS = [

    "[0.0, 2.0)",

    "[2.0, 4.0)",

    "[4.0, 6.0)",

    "[6.0, 8.0)",

    "[8.0, 10.0]"
]


# ============================================================
# 6. UTILITIES
# ============================================================

def natural_sort_key(path):
    """
    Orden natural:

        fold_1
        fold_2
        ...
        fold_10

    en lugar de:

        fold_1
        fold_10
        fold_2
    """

    text = str(path)

    return [

        int(part)
        if part.isdigit()
        else part.lower()

        for part in re.split(
            r"(\d+)",
            text
        )
    ]


# ============================================================
# 7. DISCOVER RUNS
# ============================================================

def find_prediction_runs(
    model_root,
    expected_runs=5
):
    """
    Busca recursivamente carpetas 'preds' que contengan todos
    los ficheros necesarios.

    Permite que distintos modelos tengan distinto número de runs.
    """

    if not model_root.exists():

        raise FileNotFoundError(
            "\nNo existe la ruta:\n"
            f"{model_root}"
        )

    required_files = [
        "phn_pred.npy",
        "phn_target.npy",
        "word_pred.npy",
        "word_target.npy",
        "utt_pred.npy",
        "utt_target.npy",
    ]

    runs = []

    for preds_dir in model_root.rglob("preds"):

        valid = all(
            (preds_dir / filename).exists()
            for filename in required_files
        )

        if valid:
            runs.append(preds_dir)

    runs = sorted(
        runs,
        key=natural_sort_key
    )

    # ========================================================
    # NO RUNS FOUND
    # ========================================================

    if len(runs) == 0:

        raise RuntimeError(
            f"\nERROR en {model_root}\n"
            "No se encontró ninguna ejecución válida."
        )

    # ========================================================
    # DISPLAY DISCOVERED RUNS
    # ========================================================

    print(
        f"\n  Ejecuciones válidas encontradas: "
        f"{len(runs)}"
    )

    for run in runs:
        print(
            f"    - {run}"
        )

    # ========================================================
    # CHECK THE EXPECTED COUNT
    # ========================================================

    if len(runs) != expected_runs:

        print(
            f"\n  [WARNING] Se esperaban "
            f"{expected_runs} ejecuciones, "
            f"pero se encontraron {len(runs)}."
        )

        print(
            "  Se utilizarán todas las ejecuciones "
            "válidas encontradas.\n"
        )

    return runs

# ============================================================
# 8. NORMALIZE PREDICTION AND TARGET SHAPES
# ============================================================

def normalize_prediction_target(
    pred,
    target,
    model_name,
    aspect
):
    """
    Normaliza shapes entre predicción y target.

    Casos especiales conocidos:

    GOPT:
        pred (N, ..., 1) -> squeeze

    HMamba PHONEME:
        puede guardar 52 posiciones frente a 50.

    HMamba WORD:
        algunas ejecuciones tienen una palabra extra
        en target o prediction.
        Se recorta únicamente la dimensión temporal
        al mínimo común.
    """

    pred = np.asarray(pred)
    target = np.asarray(target)

    original_pred_shape = pred.shape
    original_target_shape = target.shape

    # ========================================================
    # 1. REMOVE TRAILING SINGLETON DIMENSIONS
    # ========================================================

    while (
        pred.ndim > target.ndim
        and pred.shape[-1] == 1
    ):
        pred = np.squeeze(
            pred,
            axis=-1
        )

    while (
        target.ndim > pred.ndim
        and target.shape[-1] == 1
    ):
        target = np.squeeze(
            target,
            axis=-1
        )

    # ========================================================
    # 2. HMAMBA - PHONEME
    # ========================================================

    if (
        model_name == "HMamba"
        and aspect == "PHONEME"
    ):

        # Possible HMamba shapes:
        #
        # pred   = (2500, 52)
        # target = (2500, 50)
        #
        # The prediction and target shapes may also be reversed.

        if (
            pred.ndim == 2
            and target.ndim == 2
        ):

            common_length = min(
                pred.shape[1],
                target.shape[1],
                HMAMBA_PHN_LENGTH
            )

            if (
                pred.shape[1] != common_length
                or target.shape[1] != common_length
            ):

                print(
                    f"      HMamba PHN: "
                    f"pred {pred.shape}, "
                    f"target {target.shape} "
                    f"-> longitud {common_length}"
                )

                pred = pred[
                    :, :common_length
                ]

                target = target[
                    :, :common_length
                ]

    # ========================================================
    # 3. HMAMBA - WORD
    # ========================================================

    if (
        model_name == "HMamba"
        and aspect == "WORD"
    ):

        # Expected shapes:
        #
        # pred   -> (N, 3)
        # target -> (N, 3)
        #
        # Some HMamba runs have an extra word position
        # at the end:
        #
        # pred   -> (15966, 3)
        # target -> (15967, 3)
        #
        # Truncate to the smaller word count (original analysis behavior).

        if (
            pred.ndim == 2
            and target.ndim == 2
            and pred.shape[1] == target.shape[1]
            and pred.shape[0] != target.shape[0]
        ):

            difference = abs(
                pred.shape[0]
                - target.shape[0]
            )

            # ------------------------------------------------
            # Guard:
            # automatically handle only small
            # differences. A discrepancy of hundreds of words
            # would indicate a different problem.
            # ------------------------------------------------

            if difference <= 2:

                common_length = min(
                    pred.shape[0],
                    target.shape[0]
                )

                print(
                    f"      HMamba WORD: "
                    f"pred {pred.shape}, "
                    f"target {target.shape} "
                    f"-> recortando a "
                    f"({common_length}, {pred.shape[1]})"
                )

                pred = pred[
                    :common_length,
                    :
                ]

                target = target[
                    :common_length,
                    :
                ]

            else:

                raise ValueError(
                    "\nHMamba WORD presenta una diferencia "
                    "demasiado grande para corregirla "
                    "automáticamente:\n"
                    f"Pred:   {pred.shape}\n"
                    f"Target: {target.shape}\n"
                )

    # ========================================================
    # 4. FINAL SHAPE CHECK
    # ========================================================

    if pred.shape != target.shape:

        raise ValueError(
            f"\nShapes incompatibles\n"
            f"Model:  {model_name}\n"
            f"Aspect: {aspect}\n"
            f"Original pred:   {original_pred_shape}\n"
            f"Original target: {original_target_shape}\n"
            f"Pred final:      {pred.shape}\n"
            f"Target final:    {target.shape}\n"
        )

    return pred, target


# ============================================================
# 9. TRAIN SUPPORT
# ============================================================

def compute_train_support(
    output_list,
    aspect_name,
    gt_array,
    multiplier,
    bins,
    labels=None
):

    gt_array = np.asarray(
        gt_array
    )


    # --------------------------------------------------------
    # Remove padding.
    # --------------------------------------------------------

    valid = (

        np.isfinite(
            gt_array
        )

        &

        (
            gt_array
            >= 0
        )
    )


    gt_clean = gt_array[
        valid
    ]


    # --------------------------------------------------------
    # Original score scale.
    # --------------------------------------------------------

    gt_scaled = (
        gt_clean
        * multiplier
    )


    # ========================================================
    # BINS
    # ========================================================

    for idx, (
        min_g,
        max_g
    ) in enumerate(
        bins
    ):


        # ----------------------------------------------------
        # Interval
        # ----------------------------------------------------

        if (
            min_g == 0.0
            and
            max_g == 10.0
        ):

            mask = (

                (
                    gt_scaled
                    >= min_g
                )

                &

                (
                    gt_scaled
                    <= max_g
                )
            )


        elif max_g == 10.0:

            mask = (

                (
                    gt_scaled
                    >= min_g
                )

                &

                (
                    gt_scaled
                    <= max_g
                )
            )


        else:

            mask = (

                (
                    gt_scaled
                    >= min_g
                )

                &

                (
                    gt_scaled
                    < max_g
                )
            )


        n = int(
            np.sum(mask)
        )


        # ====================================================
        # TOTAL
        # ====================================================

        is_total = (

            min_g == 0.0

            and

            max_g == bins[-1][1]
        )


        # ====================================================
        # LABEL
        # ====================================================

        if (
            labels is not None
            and
            idx < len(labels)
        ):

            range_str = (
                labels[idx]
            )


        else:

            if (
                min_g == 0.0
                and
                max_g == 10.0
            ):

                range_str = (
                    f"[{min_g}, {max_g}]"
                )


            elif max_g == 10.0:

                range_str = (
                    f"[{min_g}, {max_g}]"
                )


            else:

                range_str = (
                    f"[{min_g}, {max_g})"
                )


        output_list.append({

            "Aspect":
                aspect_name.upper(),

            "Range":
                range_str,

            "N_Train":
                n,

            "Is_Total":
                is_total
        })


# ============================================================
# 10. BUILD THE TRAINING DISTRIBUTION
# ============================================================

def build_training_support():

    print("\n" + "=" * 90)

    print(
        "PHASE A - TRAINING SET DISTRIBUTION"
    )

    print("=" * 90)


    train_support_data = []


    # ========================================================
    # PHONEME
    # ========================================================

    phn_file = (

        TRAIN_GT_DIR
        / "tr_label_phn.npy"
    )


    if not phn_file.exists():

        raise FileNotFoundError(
            f"No existe:\n{phn_file}"
        )


    gt_phn_raw = np.load(
        phn_file
    )[:, :, 1]


    valid_phn = (
        gt_phn_raw
        != -1
    )


    gt_phn_train = (
        gt_phn_raw[
            valid_phn
        ]
        .flatten()
    )


    # ========================================================
    # WORD
    # ========================================================

    word_file = (

        TRAIN_GT_DIR
        / "tr_label_word.npy"
    )


    if not word_file.exists():

        raise FileNotFoundError(
            f"No existe:\n{word_file}"
        )


    gt_word_raw = np.load(
        word_file
    )


    word_id = (
        gt_word_raw[
            :, :, -1
        ]
    )


    word_target = (
        gt_word_raw[
            :, :, 0:3
        ]
    )


    valid_token_target = []


    for i in range(
        word_target.shape[0]
    ):

        prev_w_id = 0

        start_id = 0


        for j in range(
            word_target.shape[1]
        ):

            cur_w_id = int(
                word_id[
                    i,
                    j
                ]
            )


            if (
                cur_w_id
                != prev_w_id
            ):

                # Avoid taking means of empty arrays.
                if j > start_id:

                    token_mean = np.mean(

                        word_target[
                            i,
                            start_id:j,
                            :
                        ],

                        axis=0
                    )


                    if np.all(
                        np.isfinite(
                            token_mean
                        )
                    ):

                        valid_token_target.append(
                            token_mean
                        )


                if cur_w_id == -1:

                    break


                prev_w_id = (
                    cur_w_id
                )

                start_id = j


    gt_word_train = np.asarray(
        valid_token_target
    ).round(2)


    if gt_word_train.ndim != 2:

        raise ValueError(

            "No se ha podido construir "
            "correctamente gt_word_train. "
            f"Shape: {gt_word_train.shape}"
        )


    # ========================================================
    # UTTERANCE
    # ========================================================

    utt_file = (

        TRAIN_GT_DIR
        / "tr_label_utt.npy"
    )


    if not utt_file.exists():

        raise FileNotFoundError(
            f"No existe:\n{utt_file}"
        )


    gt_utt_train = np.load(
        utt_file
    )


    # ========================================================
    # SUPPORT
    #
    # Preserve the multipliers used
    # in the original analysis.
    # ========================================================

    compute_train_support(

        train_support_data,

        "PHONEME",

        gt_phn_train,

        multiplier=5.0,

        bins=bins_10
    )


    compute_train_support(

        train_support_data,

        "W-ACC",

        gt_word_train[:, 0],

        multiplier=1.0,

        bins=bins_10
    )


    compute_train_support(

        train_support_data,

        "W-STR",

        gt_word_train[:, 1],

        multiplier=1.0,

        bins=bins_5_10,

        labels=labels_5_10
    )


    compute_train_support(

        train_support_data,

        "W-TOT",

        gt_word_train[:, 2],

        multiplier=1.0,

        bins=bins_10
    )


    compute_train_support(

        train_support_data,

        "U-ACC",

        gt_utt_train[:, 0],

        multiplier=1.0,

        bins=bins_10
    )


    compute_train_support(

        train_support_data,

        "U-COM",

        gt_utt_train[:, 1],

        multiplier=1.0,

        bins=bins_5_10,

        labels=labels_5_10
    )


    compute_train_support(

        train_support_data,

        "U-FLU",

        gt_utt_train[:, 2],

        multiplier=1.0,

        bins=bins_10
    )


    compute_train_support(

        train_support_data,

        "U-PRO",

        gt_utt_train[:, 3],

        multiplier=1.0,

        bins=bins_10
    )


    compute_train_support(

        train_support_data,

        "U-TOT",

        gt_utt_train[:, 4],

        multiplier=1.0,

        bins=bins_10
    )


    df = pd.DataFrame(
        train_support_data
    )


    # Remove the TOTAL row.
    df = df[
        ~df[
            "Is_Total"
        ]
    ].copy()


    print(
        f"[OK] Training support: "
        f"{len(df)} bins."
    )


    return df


# ============================================================
# 11. CALCULATE MSE FOR ONE ASPECT
# ============================================================

def evaluate_mse_by_bins(
    output_list,
    model_name,
    run_idx,
    aspect_name,
    gt,
    pred,
    gt_multiplier,
    pred_multiplier,
    bins,
    labels=None
):

    gt = np.asarray(
        gt
    )

    pred = np.asarray(
        pred
    )


    if gt.shape != pred.shape:

        raise ValueError(

            f"Shape diferente antes del MSE:\n"
            f"{model_name} | {aspect_name}\n"
            f"GT:   {gt.shape}\n"
            f"Pred: {pred.shape}"
        )


    # ========================================================
    # REMOVE PADDING AND NAN VALUES
    # ========================================================

    valid = (

        np.isfinite(gt)

        &

        np.isfinite(pred)

        &

        (
            gt
            >= 0
        )
    )


    gt_clean = gt[
        valid
    ]


    pred_clean = pred[
        valid
    ]


    # ========================================================
    # SCALE TO 0-10
    # ========================================================

    gt_scaled = (
        gt_clean
        * gt_multiplier
    )


    pred_scaled = (
        pred_clean
        * pred_multiplier
    )


    # ========================================================
    # BINS
    # ========================================================

    for idx, (
        min_g,
        max_g
    ) in enumerate(
        bins
    ):


        if (
            min_g == 0.0
            and
            max_g == 10.0
        ):

            mask = (

                (
                    gt_scaled
                    >= min_g
                )

                &

                (
                    gt_scaled
                    <= max_g
                )
            )


        elif max_g == 10.0:

            mask = (

                (
                    gt_scaled
                    >= min_g
                )

                &

                (
                    gt_scaled
                    <= max_g
                )
            )


        else:

            mask = (

                (
                    gt_scaled
                    >= min_g
                )

                &

                (
                    gt_scaled
                    < max_g
                )
            )


        gt_bin = gt_scaled[
            mask
        ]


        pred_bin = pred_scaled[
            mask
        ]


        if len(gt_bin) == 0:

            mse = np.nan


        else:

            mse = float(
                np.mean(
                    (
                        pred_bin
                        - gt_bin
                    ) ** 2
                )
            )


        # ====================================================
        # LABEL
        # ====================================================

        if (
            labels is not None
            and
            idx < len(labels)
        ):

            range_str = (
                labels[idx]
            )


        else:

            if (
                min_g == 0.0
                and
                max_g == 10.0
            ):

                range_str = (
                    f"[{min_g}, {max_g}]"
                )


            elif max_g == 10.0:

                range_str = (
                    f"[{min_g}, {max_g}]"
                )


            else:

                range_str = (
                    f"[{min_g}, {max_g})"
                )


        is_total = (

            min_g == 0.0

            and

            max_g == bins[-1][1]
        )


        output_list.append({

            "Model":
                model_name,

            "Run":
                run_idx,

            "Aspect":
                aspect_name.upper(),

            "Range":
                range_str,

            "MSE":
                mse,

            "Is_Total":
                is_total,
        })


# ============================================================
# 12. EVALUATE ORIGINAL OR BOMA
# ============================================================

def evaluate_condition(
    condition
):

    print("\n" + "#" * 90)

    print(
        f"PHASE B - {condition.upper()}"
    )

    print("#" * 90)


    error_data = []


    # ========================================================
    # 6 ARCHITECTURES
    # ========================================================

    for (
        model_name,
        model_paths
    ) in MODELS.items():


        model_root = (
            model_paths[
                condition
            ]
        )


        print(
            "\n"
            + "-" * 90
        )

        print(
            f"{model_name} | "
            f"{condition.upper()}"
        )

        print(
            f"Root: {model_root}"
        )


                # ============================================================
        # EXPECTED RUN COUNT FOR THIS MODEL AND CONDITION
        # ============================================================

        run_key = f"n_runs_{condition}"

        expected_runs = model_paths.get(
            run_key,
            N_RUNS
        )

        runs = find_prediction_runs(
            model_root,
            expected_runs
        )

        print(
            f"  -> Se utilizarán {len(runs)} "
            f"ejecuciones para {model_name} "
            f"({condition.upper()})"
        )


        # ====================================================
        # 5 RUNS
        # ====================================================

        for run_idx, preds_dir in enumerate(
            runs,
            start=1
        ):

            print(
                f"  Run {run_idx}: "
                f"{preds_dir}"
            )


            # =================================================
            # LOAD PHONEME
            # =================================================

            phn_pred = np.load(

                preds_dir
                / "phn_pred.npy"
            )


            phn_target = np.load(

                preds_dir
                / "phn_target.npy"
            )


            phn_pred, phn_target = (
                normalize_prediction_target(

                    phn_pred,

                    phn_target,

                    model_name,

                    "PHONEME"
                )
            )


            # =================================================
            # LOAD WORD
            # =================================================

            word_pred = np.load(

                preds_dir
                / "word_pred.npy"
            )


            word_target = np.load(

                preds_dir
                / "word_target.npy"
            )


            word_pred, word_target = (
                normalize_prediction_target(

                    word_pred,

                    word_target,

                    model_name,

                    "WORD"
                )
            )


            # =================================================
            # LOAD UTTERANCE
            # =================================================

            utt_pred = np.load(

                preds_dir
                / "utt_pred.npy"
            )


            utt_target = np.load(

                preds_dir
                / "utt_target.npy"
            )


            utt_pred, utt_target = (
                normalize_prediction_target(

                    utt_pred,

                    utt_target,

                    model_name,

                    "UTTERANCE"
                )
            )


            # =================================================
            # PHONEME
            # =================================================

            evaluate_mse_by_bins(

                error_data,

                model_name,

                run_idx,

                "PHONEME",

                phn_target.flatten(),

                phn_pred.flatten(),

                gt_multiplier=5.0,

                pred_multiplier=5.0,

                bins=bins_10
            )


            # =================================================
            # WORD
            # =================================================

            evaluate_mse_by_bins(

                error_data,

                model_name,

                run_idx,

                "W-ACC",

                word_target[:, 0],

                word_pred[:, 0],

                gt_multiplier=5.0,

                pred_multiplier=5.0,

                bins=bins_10
            )


            evaluate_mse_by_bins(

                error_data,

                model_name,

                run_idx,

                "W-STR",

                word_target[:, 1],

                word_pred[:, 1],

                gt_multiplier=5.0,

                pred_multiplier=5.0,

                bins=bins_5_10,

                labels=labels_5_10
            )


            evaluate_mse_by_bins(

                error_data,

                model_name,

                run_idx,

                "W-TOT",

                word_target[:, 2],

                word_pred[:, 2],

                gt_multiplier=5.0,

                pred_multiplier=5.0,

                bins=bins_10
            )


            # =================================================
            # UTTERANCE
            # =================================================

            evaluate_mse_by_bins(

                error_data,

                model_name,

                run_idx,

                "U-ACC",

                utt_target[:, 0],

                utt_pred[:, 0],

                gt_multiplier=5.0,

                pred_multiplier=5.0,

                bins=bins_10
            )


            evaluate_mse_by_bins(

                error_data,

                model_name,

                run_idx,

                "U-COM",

                utt_target[:, 1],

                utt_pred[:, 1],

                gt_multiplier=5.0,

                pred_multiplier=5.0,

                bins=bins_5_10,

                labels=labels_5_10
            )


            evaluate_mse_by_bins(

                error_data,

                model_name,

                run_idx,

                "U-FLU",

                utt_target[:, 2],

                utt_pred[:, 2],

                gt_multiplier=5.0,

                pred_multiplier=5.0,

                bins=bins_10
            )


            evaluate_mse_by_bins(

                error_data,

                model_name,

                run_idx,

                "U-PRO",

                utt_target[:, 3],

                utt_pred[:, 3],

                gt_multiplier=5.0,

                pred_multiplier=5.0,

                bins=bins_10
            )


            evaluate_mse_by_bins(

                error_data,

                model_name,

                run_idx,

                "U-TOT",

                utt_target[:, 4],

                utt_pred[:, 4],

                gt_multiplier=5.0,

                pred_multiplier=5.0,

                bins=bins_10
            )


    # ========================================================
    # DATAFRAME
    # ========================================================

    df_error = pd.DataFrame(
        error_data
    )


    # Remove the TOTAL row.
    df_error = df_error[

        ~df_error[
            "Is_Total"
        ]

    ].copy()


    # ========================================================
    # STEP 1:
    #
    # AVERAGE THE FIVE RUNS WITHIN EACH ARCHITECTURE
    # ========================================================

    df_model = (

        df_error

        .groupby(

            [
                "Model",
                "Aspect",
                "Range"
            ],

            as_index=False
        )

        .agg(

            MSE_model=(
                "MSE",
                "mean"
            )
        )
    )


    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    model_counts = (

        df_model

        .groupby(
            [
                "Aspect",
                "Range"
            ]
        )[
            "Model"
        ]

        .nunique()
    )


    problematic = (
        model_counts[
            model_counts
            != len(MODELS)
        ]
    )


    if len(problematic) > 0:

        print(
            "\n[WARNING] Algunos bins "
            "no tienen resultados para los "
            "6 modelos:"
        )

        print(
            problematic
        )


    # ========================================================
    # STEP 2:
    #
    # AVERAGE ACROSS THE SIX ARCHITECTURES
    #
    # Each architecture receives weight 1/6.
    # ========================================================

    df_condition = (

        df_model

        .groupby(

            [
                "Aspect",
                "Range"
            ],

            as_index=False
        )

        .agg(

            MSE_mean=(
                "MSE_model",
                "mean"
            ),

            MSE_std=(
                "MSE_model",
                "std"
            ),

            N_models=(
                "Model",
                "nunique"
            )
        )
    )


    print(
        "\n[OK] "
        f"{condition.upper()} evaluado."
    )


    return df_condition


# ============================================================
# 13. PREPARE HEATMAP MATRICES
# ============================================================

def prepare_heatmap_matrices(
    df_support,
    df_condition
):

    df_final = pd.merge(

        df_support,

        df_condition,

        on=[
            "Aspect",
            "Range"
        ],

        how="inner"
    )


    ordered_aspects = sorted(

        df_final[
            "Aspect"
        ].unique()
    )


    # ========================================================
    # N TRAIN
    # ========================================================

    n_train = (

        df_final

        .pivot(

            index="Aspect",

            columns="Range",

            values="N_Train"
        )

        .reindex(

            index=ordered_aspects,

            columns=ORDERED_COLS
        )

        .fillna(0)
    )


    # ========================================================
    # MSE MEAN
    # ========================================================

    mse_mean = (

        df_final

        .pivot(

            index="Aspect",

            columns="Range",

            values="MSE_mean"
        )

        .reindex(

            index=ordered_aspects,

            columns=ORDERED_COLS
        )
    )


    # ========================================================
    # MSE STANDARD DEVIATION ACROSS ARCHITECTURES
    # ========================================================

    mse_std = (

        df_final

        .pivot(

            index="Aspect",

            columns="Range",

            values="MSE_std"
        )

        .reindex(

            index=ordered_aspects,

            columns=ORDERED_COLS
        )
    )


    return (
        n_train,
        mse_mean,
        mse_std
    )


# ============================================================
# 14. GENERATE THE HEATMAP
# ============================================================

def create_heatmap(
    condition,
    df_support,
    df_condition
):

    (
        n_train,
        mse_mean,
        mse_std
    ) = prepare_heatmap_matrices(

        df_support,

        df_condition
    )


    # ========================================================
    # ANNOTATIONS
    # ========================================================

    annot_n = (

        n_train
        .copy()
        .astype(object)
    )


    annot_mse = (

        mse_mean
        .copy()
        .astype(object)
    )


    for row in n_train.index:

        for col in n_train.columns:


            # ------------------------------------------------
            # TRAIN N
            # ------------------------------------------------

            n = (
                n_train
                .loc[
                    row,
                    col
                ]
            )


            annot_n.loc[
                row,
                col
            ] = (

                "0"

                if n == 0

                else f"{n:.0f}"
            )


            # ------------------------------------------------
            # MSE
            # ------------------------------------------------

            mean = (
                mse_mean
                .loc[
                    row,
                    col
                ]
            )


            std = (
                mse_std
                .loc[
                    row,
                    col
                ]
            )


            if pd.isna(mean):

                annot_mse.loc[
                    row,
                    col
                ] = "---"


            elif pd.isna(std):

                annot_mse.loc[
                    row,
                    col
                ] = (
                    f"{mean:.2f}"
                )


            else:

                annot_mse.loc[
                    row,
                    col
                ] = (

                    f"{mean:.2f}"
                    "\n"
                    f"±{std:.2f}"
                )


    # ========================================================
    # FIGURE
    # ========================================================

    sns.set_theme(
        style="white"
    )


    fig, axes = plt.subplots(

        1,
        2,

        figsize=(
            18,
            8
        ),

        gridspec_kw={

            "width_ratios":
                [1, 1.15]
        }
    )


    axes[0].set_facecolor(
        "#f2f2f2"
    )


    axes[1].set_facecolor(
        "#f2f2f2"
    )


    # ========================================================
    # LEFT PANEL:
    #
    # TRAIN DISTRIBUTION
    # ========================================================

    sns.heatmap(

        n_train,

        annot=annot_n,

        fmt="",

        cmap="Greys",

        norm=SymLogNorm(

            linthresh=1,

            vmin=0,

            base=10
        ),

        cbar_kws={

            "label":
                "Train Support N "
                "(SymLog Scale)"
        },

        ax=axes[0],

        linewidths=0.5
    )


    axes[0].set_title(

        "Training Set Distribution (N)",

        fontweight="bold",

        pad=15
    )


    axes[0].set_xlabel(

        "Score Range",

        fontweight="bold"
    )


    axes[0].set_ylabel(

        "Acoustic Aspect",

        fontweight="bold"
    )


    # ========================================================
    # RIGHT PANEL:
    #
    # MEAN MSE
    # ========================================================

    sns.heatmap(

        mse_mean,

        annot=annot_mse,

        fmt="",

        cmap="Reds",

        # NOTE:
        # ORIGINAL and BOMA use exactly
        # the same scale.
        vmin=0,

        vmax=MSE_VMAX,

        cbar_kws={

            "label":
                "Mean Test MSE "
                "Across Models"
        },

        ax=axes[1],

        linewidths=0.5
    )


    axes[1].set_title(

        (
            f"{condition.upper()} — "
            "Prediction Error\n"
            "Mean MSE ± Std Across 6 Models"
        ),

        fontweight="bold",

        pad=15
    )


    axes[1].set_xlabel(

        "Score Range",

        fontweight="bold"
    )


    axes[1].set_ylabel(
        ""
    )


    # ========================================================
    # SAVE
    # ========================================================

    plt.tight_layout()


    output_file = (

        f"mse_vs_n_all_models_"
        f"{condition.upper()}.png"
    )


    plt.savefig(

        output_file,

        dpi=300,

        bbox_inches="tight"
    )


    plt.close(
        fig
    )


    print(
        f"[OK] Creado: "
        f"{output_file}"
    )


# ============================================================
# 15. PRINT THE SUMMARY TABLE
# ============================================================

def print_condition_summary(
    condition,
    df_condition
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"SUMMARY - "
        f"{condition.upper()}"
    )

    print(
        "=" * 100
    )


    with pd.option_context(

        "display.max_rows",
        None,

        "display.max_columns",
        None,

        "display.width",
        200
    ):

        print(

            df_condition
            .sort_values(
                [
                    "Aspect",
                    "Range"
                ]
            )

            .to_string(
                index=False
            )
        )


# ============================================================
# 16. MAIN
# ============================================================

def main():

    print(
        "\n"
        + "#" * 100
    )

    print(
        "MEAN MODEL HEATMAP ANALYSIS"
    )

    print(
        "#" * 100
    )


    # ========================================================
    # TRAIN SUPPORT
    # ========================================================

    df_support = (
        build_training_support()
    )


    # ========================================================
    # ORIGINAL
    # ========================================================

    df_original = (
        evaluate_condition(
            "original"
        )
    )


    # ========================================================
    # BOMA
    # ========================================================

    df_boma = (
        evaluate_condition(
            "boma"
        )
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print_condition_summary(

        "original",

        df_original
    )


    print_condition_summary(

        "boma",

        df_boma
    )


    # ========================================================
    # HEATMAP ORIGINAL
    # ========================================================

    create_heatmap(

        "original",

        df_support,

        df_original
    )


    # ========================================================
    # HEATMAP BOMA
    # ========================================================

    create_heatmap(

        "boma",

        df_support,

        df_boma
    )


    # ========================================================
    # END
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "ARCHIVOS GENERADOS"
    )

    print(
        "=" * 100
    )

    print(
        "1) mse_vs_n_all_models_ORIGINAL.png"
    )

    print(
        "2) mse_vs_n_all_models_BOMA.png"
    )

    print(
        "=" * 100
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()