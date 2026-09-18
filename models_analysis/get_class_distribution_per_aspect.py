# Archived analysis implementation; paths and metric conventions are preserved.
# See tools/evaluate_saved.py for evaluation with an explicit input and output.
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA LOADING (Continuous Mean Ground Truth Only)
# ==========================================
dir = "../06_M3C/data/GoP_VC_Features_Embbeding_norm"

# Load Continuous Ground Truth
gt_phn = np.load(f"{dir}/te_label_phn.npy")[:, :, 1]

# Mask application to discard padding or null values
valid_mask = gt_phn != -1
gt_phn = gt_phn[valid_mask].flatten()

gt_word = np.load(f"{dir}/te_label_word.npy")

word_id = gt_word[:, :, -1]
target = gt_word[:, :, 0:3]

valid_token_target = []

# The nested loops preserve the original word-aggregation procedure.
for i in range(target.shape[0]):
    prev_w_id = 0
    start_id = 0
    
    for j in range(target.shape[1]):
        cur_w_id = int(word_id[i, j])
        
        # Detect a change in word ID.
        if cur_w_id != prev_w_id:
            # Extract and average the phoneme block for this word.
            valid_token_target.append(np.mean(target[i, start_id:j, :], axis=0))
            
            # Stop at padding.
            if cur_w_id == -1: 
                break
            else: 
                prev_w_id = cur_w_id
                start_id = j

gt_word = np.array(valid_token_target).round(2)
print(f"gt_word shape: {gt_word.shape}")

gt_utt = np.load(f"{dir}/te_label_utt.npy")
print(f"gt_utt shape: {gt_utt.shape}")

# ==========================================
# 2. GROUPING AND REPORTING FUNCTION
# ==========================================
def print_imbalance_table(name, data, bins, labels, total_label):
    """
    Calculates and prints sample count per interval and the Majority Class Percentage (%MC).
    """
    clean_data = data[~np.isnan(data)]
    total = len(clean_data)
    counts, _ = np.histogram(clean_data, bins=bins)
    
    # Majority Class Percentage (%MC)
    mc_pct = (np.max(counts) / total * 100.0) if total > 0 else 0.0

    # Report Output
    print(f"{name.center(28)}")
    print(f"{'RANGE':<10} | {'N'}")
    print("-" * 28)
    for i in range(len(counts)):
        print(f"{labels[i]:<10} | {counts[i]}")
    print("-" * 28)
    print(f"{total_label:<10} | {total}")
    print(f"{'% MC':<10} | {mc_pct:.2f}%\n")

# ==========================================
# 3. IMBALANCE REPORT GENERATION
# ==========================================
print("="*80)
print(" 1. IMBALANCE EVIDENCE (GROUND TRUTH: CONTINUOUS MEAN)")
print("="*80 + "\n")

# --- A. PHONEME LEVEL (Scale 0.0 - 2.0) ---
bins_phn = [0.0, 2.0, 4.0, 6.0, 8.0, 10.001]
labels_phn = ['0-2', '2-4', '4-6', '6-8', '8-10']

print_imbalance_table("Phoneme", gt_phn * 5.0, bins_phn, labels_phn, "0-10")


# --- B. WORD LEVEL ---
bins_10 = [0.0, 2.0, 4.0, 6.0, 8.0, 10.001]
labels_10 = ['0-2', '2-4', '4-6', '6-8', '8-10']

bins_5_10 = [0.0, 7.5, 10.001]
labels_5_10 = ['5', '10']

# W-ACC
print_imbalance_table("W-ACC", gt_word[:, 0], bins_10, labels_10, "0-10")

# W-STRESS 
print_imbalance_table("W-STRESS", gt_word[:, 1], bins_5_10, labels_5_10, "5 / 10")

# W-TOTAL 
print_imbalance_table("W-TOTAL", gt_word[:, 2], bins_10, labels_10, "0-10")

# --- C. UTTERANCE LEVEL ---
# U-ACC 
print_imbalance_table("U-ACC", gt_utt[:, 0], bins_10, labels_10, "0-10")

# U-COM 
print_imbalance_table("U-COM", gt_utt[:, 1], bins_5_10, labels_5_10, "5 / 10")

# U-FLU 
print_imbalance_table("U-FLU", gt_utt[:, 2], bins_10, labels_10, "0-10")

# U-PRO 
print_imbalance_table("U-PRO", gt_utt[:, 3], bins_10, labels_10, "0-10")

# U-TOT 
print_imbalance_table("U-TOT", gt_utt[:, 4], bins_10, labels_10, "0-10")

# ==========================================
# 4. HISTOGRAM GENERATION
# ==========================================
import matplotlib.pyplot as plt


def plot_aspect_histogram(aspect):
    """
    Generates the histogram for the selected assessment aspect.

    Available aspects:
        PHN
        W-ACC
        W-STRESS
        W-TOTAL
        U-ACC
        U-COM
        U-FLU
        U-PRO
        U-TOT
    """

    # ------------------------------------------
    # Aspect configuration
    # ------------------------------------------
    aspect_data = {
        "PHN": {
            "data": gt_phn * 5.0,
            "bins": bins_phn,
            "xlabel": "Phoneme score",
            "title": "Phoneme Score Distribution"
        },

        "W-ACC": {
            "data": gt_word[:, 0],
            "bins": bins_10,
            "xlabel": "Word Accuracy Score",
            "title": "Word Accuracy Distribution"
        },

        "W-STRESS": {
            "data": gt_word[:, 1],
            "bins": bins_5_10,
            "xlabel": "Word Stress Score",
            "title": "Word Stress Distribution"
        },

        "W-TOTAL": {
            "data": gt_word[:, 2],
            "bins": bins_10,
            "xlabel": "Word Total Score",
            "title": "Word Total Distribution"
        },

        "U-ACC": {
            "data": gt_utt[:, 0],
            "bins": bins_10,
            "xlabel": "Utterance Accuracy Score",
            "title": "Utterance Accuracy Distribution"
        },

        "U-COM": {
            "data": gt_utt[:, 1],
            "bins": bins_5_10,
            "xlabel": "Utterance Completeness Score",
            "title": "Utterance Completeness Distribution"
        },

        "U-FLU": {
            "data": gt_utt[:, 2],
            "bins": bins_10,
            "xlabel": "Utterance Fluency Score",
            "title": "Utterance Fluency Distribution"
        },

        "U-PRO": {
            "data": gt_utt[:, 3],
            "bins": bins_10,
            "xlabel": "Utterance Prosody Score",
            "title": "Utterance Prosody Distribution"
        },

        "U-TOT": {
            "data": gt_utt[:, 4],
            "bins": bins_10,
            "xlabel": "Utterance Total Score",
            "title": "Utterance Total Distribution"
        }
    }

    aspect = aspect.upper()

    if aspect not in aspect_data:
        raise ValueError(
            f"Unknown aspect '{aspect}'. "
            f"Available aspects: {list(aspect_data.keys())}"
        )

    config = aspect_data[aspect]

    data = config["data"]
    bins = config["bins"]

    # Remove NaNs
    data = data[~np.isnan(data)]

    # ------------------------------------------
    # Plot
    # ------------------------------------------
    # Percentiles
    # ------------------------------------------
    p33 = np.percentile(data, 33)
    p66 = np.percentile(data, 66)

    # ------------------------------------------
    # Plot
    # ------------------------------------------
    plt.figure(figsize=(10, 5.5))

    plt.hist(
        data,
        bins=np.arange(0, 11, 1),
        color="#E15759",        # Use the red color from the figure.
        edgecolor="black",
        linewidth=1.2,
        alpha=1.0
    )

    # Percentile lines
    plt.axvline(
        p33,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"P33: {p33:.2f}"
    )

    plt.axvline(
        p66,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"P66: {p66:.2f}"
    )

    # Labels and title
    plt.xlabel("Score Value", fontsize=14)
    plt.ylabel("Count", fontsize=14)

    plt.title(
        config["title"],
        fontsize=18,
        fontweight="bold",
        pad=12
    )

    # X axis from 0 to 10, step 1
    plt.xticks(
        np.arange(0, 11, 1),
        fontsize=12
    )

    plt.yticks(fontsize=12)

    plt.xlim(0, 10)

    # Horizontal grid
    plt.grid(
        axis="y",
        linestyle="--",
        linewidth=1,
        alpha=0.5
    )

    # Percentile legend
    plt.legend(
        loc="upper left",
        fontsize=12,
        frameon=True
    )

    plt.tight_layout()

    # Save
    plt.savefig(
        f"histogram_{aspect}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.savefig(f"histogram_{aspect}.png", dpi=300, bbox_inches="tight")


# ==========================================
# SELECT ASPECT TO PLOT
# ==========================================

ASPECT_TO_PLOT = "U-TOT"

plot_aspect_histogram(ASPECT_TO_PLOT)
