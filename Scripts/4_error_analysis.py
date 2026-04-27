# ============================================================
# 4_error_analysis.py
#
# PURPOSE:
# This is the final script — the "why" layer.
# It looks at where models failed and identifies patterns.
#
# It answers:
#   - Which relation types are most commonly confused?
#   - Which combination had the most failures?
#   - Which papers did EVERY combination get wrong?
#   - What patterns exist in the abstracts of failed papers?
#
# It generates confusion matrices for every combination
# and for the ensemble — showing exactly which labels
# got confused with which other labels.
#
# INPUT FILES:
#   - data/dataset_clean.xlsx
#   - results/ensemble_predictions.csv   (from Script 3)
#   - results/scores_overall.csv         (from Script 1)
#
# OUTPUT FILES (saved to results/):
#   - failed_papers.csv               → papers ensemble got wrong
#   - all_combos_failed.csv           → papers ALL 9 combos got wrong
#   - error_summary.csv               → error counts per combination
#   - confusion_matrices/             → folder of confusion matrix PNGs
#       ensemble_causal_cm.png
#       ensemble_relation_cm.png
#       [combo]_causal_cm.png         → one per combination
#       [combo]_relation_cm.png
#   - heatmap_causal_errors.png       → which combos fail on which labels
#   - heatmap_relation_errors.png
# ============================================================


# ── IMPORTS ─────────────────────────────────────────────────

import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from collections import Counter


# ── CONFIGURATION ────────────────────────────────────────────

DATA_PATH       = "data/dataset_clean.xlsx"
PRED_PATH       = "results/ensemble_predictions.csv"
SCORES_PATH     = "results/scores_overall.csv"
RESULTS_PATH    = "results/"
CM_PATH         = "results/confusion_matrices/"

GT_CAUSAL   = "manual_causal"
GT_RELATION = "manual_relation_type"

DISCIPLINES = [
    "Biomedical Engineering",
    "Molecular Biology / Genetics",
    "Cancer Biology",
    "Chemistry"
]

COMBINATIONS = [
    {"name": "GPT4 Zero-Shot",    "causal": "gpt4_zeroshot_causal",    "relation": "gpt4_zeroshot_relation"},
    {"name": "GPT4 Few-Shot",     "causal": "gpt4_fewshot_causal",     "relation": "gpt4_fewshot_relation"},
    {"name": "GPT4 Role-Based",   "causal": "gpt4_rolebased_causal",   "relation": "gpt4_rolebased_relation"},
    {"name": "Claude Zero-Shot",  "causal": "claude_zeroshot_causal",  "relation": "claude_zeroshot_relation"},
    {"name": "Claude Few-Shot",   "causal": "claude_fewshot_causal",   "relation": "claude_fewshot_relation"},
    {"name": "Claude Role-Based", "causal": "claude_rolebased_causal", "relation": "claude_rolebased_relation"},
    {"name": "Gemini Zero-Shot",  "causal": "gemini_zeroshot_causal",  "relation": "gemini_zeroshot_relation"},
    {"name": "Gemini Few-Shot",   "causal": "gemini_fewshot_causal",   "relation": "gemini_fewshot_relation"},
    {"name": "Gemini Role-Based", "causal": "gemini_rolebased_causal", "relation": "gemini_rolebased_relation"},
]

# Causal labels
CAUSAL_LABELS = ["yes", "no"]

# Relation type labels — ordered by frequency in ground truth
RELATION_LABELS = [
    "measurement/method",
    "mechanistic/theory",
    "description",
    "association",
    "correlation",
    "hypothesis"
]


# ── HELPER FUNCTIONS ─────────────────────────────────────────

def build_confusion_matrix(y_true, y_pred, labels):
    """
    Builds a confusion matrix as a 2D list.

    Rows    = actual labels (ground truth)
    Columns = predicted labels (what the model said)
    Cell    = how many times that true→predicted pair occurred

    A perfect model has all values on the diagonal (top-left to bottom-right)
    and zeros everywhere else.

    Parameters:
        y_true  → list of ground truth labels
        y_pred  → list of predicted labels
        labels  → ordered list of all possible labels

    Returns:
        matrix  → 2D list (rows = true, cols = predicted)
        valid_labels → labels that actually appear in the data
    """
    # Filter to only valid pairs (both values exist and are known labels)
    pairs = [
        (t, p) for t, p in zip(y_true, y_pred)
        if pd.notna(t) and pd.notna(p)
        and t in labels and p in labels
    ]

    if len(pairs) < 2:
        return None, []

    true_vals, pred_vals = zip(*pairs)

    # Only include labels that actually appear in this data
    valid_labels = [l for l in labels if l in set(true_vals) or l in set(pred_vals)]

    n = len(valid_labels)
    label_idx = {l: i for i, l in enumerate(valid_labels)}
    # label_idx maps each label to its row/column index

    # Initialize empty matrix with zeros
    matrix = [[0] * n for _ in range(n)]

    # Fill matrix — each pair increments one cell
    for t, p in zip(true_vals, pred_vals):
        if t in label_idx and p in label_idx:
            row = label_idx[t]   # row = actual label
            col = label_idx[p]   # col = predicted label
            matrix[row][col] += 1

    return matrix, valid_labels


def plot_confusion_matrix(matrix, labels, title, save_path):
    """
    Draws and saves a confusion matrix as a heatmap image.

    The diagonal shows correct predictions (darker = more correct).
    Off-diagonal shows errors (how often true label X was predicted as Y).
    Numbers inside each cell show the count.
    """
    if matrix is None or len(labels) == 0:
        print(f"    Skipped (not enough data): {title}")
        return

    n = len(labels)
    mat = np.array(matrix)

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))

    # Use a blue colormap — darker = higher count
    im = ax.imshow(mat, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Set tick labels on both axes
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Add count numbers inside each cell
    # Use white text on dark cells, black text on light cells
    thresh = mat.max() / 2.0
    for i in range(n):
        for j in range(n):
            color = "white" if mat[i, j] > thresh else "black"
            ax.text(
                j, i,
                str(mat[i, j]),
                ha="center", va="center",
                color=color, fontsize=10, fontweight="bold"
            )

    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("Actual Label (Ground Truth)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ── MAIN FUNCTION ────────────────────────────────────────────

def run_error_analysis():

    # Load all required files
    print("Loading datasets...")
    df      = pd.read_excel(DATA_PATH)
    pred_df = pd.read_csv(PRED_PATH)

    print(f"  Dataset            : {len(df)} rows")
    print(f"  Ensemble predictions: {len(pred_df)} rows")
    print()

    # Create output folders
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(CM_PATH, exist_ok=True)


    # ── STEP 1: FAILED PAPERS ANALYSIS ───────────────────────
    # Find which papers the ensemble got wrong

    print("=" * 60)
    print("STEP 1: FAILED PAPERS ANALYSIS")
    print("=" * 60)

    # Papers where ensemble got causal prediction wrong
    failed_causal = pred_df[pred_df["causal_correct"] == False].copy()

    # Papers where causal=no AND relation prediction wrong
    failed_relation = pred_df[
        (pred_df["manual_causal"] == "no") &
        (pred_df["relation_correct"] == False)
    ].copy()

    print(f"\n  Ensemble wrong on causal task    : {len(failed_causal)} / {len(pred_df)} papers")
    print(f"  Ensemble wrong on relation task  : {len(failed_relation)} / {pred_df['manual_causal'].eq('no').sum()} papers")

    # Save failed papers with full details for manual review
    failed_all = pred_df[
        (pred_df["causal_correct"] == False) |
        (pred_df["relation_correct"] == False)
    ].copy()

    failed_all.to_csv(RESULTS_PATH + "failed_papers.csv", index=False)
    print(f"  Saved: failed_papers.csv ({len(failed_all)} papers)")

    # Show breakdown by discipline
    print(f"\n  Failed papers by discipline:")
    for disc in DISCIPLINES:
        disc_fail = failed_all[failed_all["discipline"] == disc]
        disc_total = pred_df[pred_df["discipline"] == disc]
        print(f"    {disc:<38}: {len(disc_fail)} / {len(disc_total)} failed")


    # ── STEP 2: PAPERS ALL 9 COMBOS GOT WRONG ────────────────
    # These are systematic failures — hardest papers for LLMs

    print()
    print("=" * 60)
    print("STEP 2: PAPERS ALL 9 COMBINATIONS GOT WRONG (Systematic Failures)")
    print("=" * 60)

    all_wrong_rows = []

    for idx, row in df.iterrows():

        if pd.isna(row[GT_CAUSAL]):
            continue

        # Check if ALL 9 combinations got causal wrong
        all_causal_wrong = all(
            pd.notna(row[combo["causal"]]) and row[combo["causal"]] != row[GT_CAUSAL]
            for combo in COMBINATIONS
        )

        if all_causal_wrong:
            all_wrong_rows.append({
                "num"               : row["num"],
                "title"             : row["title"],
                "discipline"        : row["discipline"],
                "manual_causal"     : row[GT_CAUSAL],
                "manual_relation"   : row[GT_RELATION],
                "abstract_snippet"  : str(row["abstract"])[:300] + "..."
                # First 300 chars of abstract for quick review
            })

    all_wrong_df = pd.DataFrame(all_wrong_rows)
    all_wrong_df.to_csv(RESULTS_PATH + "all_combos_failed.csv", index=False)

    print(f"\n  Papers ALL 9 combos got wrong: {len(all_wrong_df)}")
    print(f"  Saved: all_combos_failed.csv")

    if len(all_wrong_df) > 0:
        print(f"\n  Discipline breakdown of systematic failures:")
        for disc in DISCIPLINES:
            count = (all_wrong_df["discipline"] == disc).sum()
            if count > 0:
                print(f"    {disc}: {count} papers")


    # ── STEP 3: ERROR COUNT PER COMBINATION ──────────────────
    # Count how many papers each combination got wrong

    print()
    print("=" * 60)
    print("STEP 3: ERROR COUNT PER COMBINATION")
    print("=" * 60)

    error_summary = []

    for combo in COMBINATIONS:

        # Causal errors — rows where prediction != ground truth
        valid_causal = df[
            df[GT_CAUSAL].notna() & df[combo["causal"]].notna()
        ]
        causal_errors = (
            valid_causal[GT_CAUSAL] != valid_causal[combo["causal"]]
        ).sum()

        # Relation errors — only on "no" rows
        no_rows = df[df[GT_CAUSAL] == "no"]
        valid_relation = no_rows[
            no_rows[GT_RELATION].notna() & no_rows[combo["relation"]].notna()
        ]
        relation_errors = (
            valid_relation[GT_RELATION] != valid_relation[combo["relation"]]
        ).sum()

        total_causal   = len(valid_causal)
        total_relation = len(valid_relation)

        error_summary.append({
            "Combination"      : combo["name"],
            "Causal_Errors"    : causal_errors,
            "Causal_Total"     : total_causal,
            "Causal_Error_Rate": round(causal_errors / total_causal, 4) if total_causal > 0 else 0,
            "Relation_Errors"  : relation_errors,
            "Relation_Total"   : total_relation,
            "Relation_Error_Rate": round(relation_errors / total_relation, 4) if total_relation > 0 else 0,
        })

        print(f"  {combo['name']:<24} | Causal errors: {causal_errors}/{total_causal} ({causal_errors/total_causal*100:.1f}%) | Relation errors: {relation_errors}/{total_relation} ({relation_errors/total_relation*100:.1f}%)")

    error_df = pd.DataFrame(error_summary)
    error_df = error_df.sort_values("Causal_Error_Rate")
    error_df.to_csv(RESULTS_PATH + "error_summary.csv", index=False)
    print(f"\n  Saved: error_summary.csv")


    # ── STEP 4: CONFUSION MATRICES ────────────────────────────
    # Generate one confusion matrix per combination + ensemble

    print()
    print("=" * 60)
    print("STEP 4: GENERATING CONFUSION MATRICES")
    print("=" * 60)

    # Ensemble confusion matrices
    print("\n  Ensemble confusion matrices...")

    # Causal
    cm, labels = build_confusion_matrix(
        pred_df[GT_CAUSAL],
        pred_df["ensemble_causal"],
        CAUSAL_LABELS
    )
    plot_confusion_matrix(
        cm, labels,
        "Ensemble — Causal Classification\n(yes / no)",
        CM_PATH + "ensemble_causal_cm.png"
    )
    print("    Saved: ensemble_causal_cm.png")

    # Relation
    no_pred = pred_df[pred_df[GT_CAUSAL] == "no"]
    cm, labels = build_confusion_matrix(
        no_pred[GT_RELATION],
        no_pred["ensemble_relation"],
        RELATION_LABELS
    )
    plot_confusion_matrix(
        cm, labels,
        "Ensemble — Relation Type Classification",
        CM_PATH + "ensemble_relation_cm.png"
    )
    print("    Saved: ensemble_relation_cm.png")

    # Individual combination confusion matrices
    print("\n  Individual combination confusion matrices...")

    for combo in COMBINATIONS:
        safe_name = combo["name"].replace(" ", "_").replace("-", "")

        # Causal CM
        cm, labels = build_confusion_matrix(
            df[GT_CAUSAL],
            df[combo["causal"]],
            CAUSAL_LABELS
        )
        plot_confusion_matrix(
            cm, labels,
            f"{combo['name']} — Causal Classification",
            CM_PATH + f"{safe_name}_causal_cm.png"
        )

        # Relation CM
        no_df = df[df[GT_CAUSAL] == "no"]
        cm, labels = build_confusion_matrix(
            no_df[GT_RELATION],
            no_df[combo["relation"]],
            RELATION_LABELS
        )
        plot_confusion_matrix(
            cm, labels,
            f"{combo['name']} — Relation Type Classification",
            CM_PATH + f"{safe_name}_relation_cm.png"
        )

        print(f"    Saved: {safe_name}_causal_cm.png + _relation_cm.png")


    # ── STEP 5: ERROR HEATMAPS ────────────────────────────────
    # Summary heatmap — which combination fails most on which label

    print()
    print("=" * 60)
    print("STEP 5: ERROR HEATMAPS")
    print("=" * 60)

    combo_names = [c["name"] for c in COMBINATIONS]

    # Causal error heatmap
    # Rows = combinations, Cols = true label (yes/no)
    # Cell = how many times that combo got that label wrong
    causal_errors_matrix = []

    for combo in COMBINATIONS:
        row_errors = []
        for true_label in CAUSAL_LABELS:
            subset = df[df[GT_CAUSAL] == true_label]
            errors = (
                subset[combo["causal"]].notna() &
                (subset[combo["causal"]] != true_label)
            ).sum()
            row_errors.append(errors)
        causal_errors_matrix.append(row_errors)

    fig, ax = plt.subplots(figsize=(6, 8))
    mat = np.array(causal_errors_matrix)
    im  = ax.imshow(mat, cmap="Reds", aspect="auto")
    plt.colorbar(im, ax=ax, label="Number of Errors")

    ax.set_xticks(range(len(CAUSAL_LABELS)))
    ax.set_xticklabels(CAUSAL_LABELS, fontsize=10)
    ax.set_yticks(range(len(combo_names)))
    ax.set_yticklabels(combo_names, fontsize=9)

    # Add counts inside cells
    for i in range(len(combo_names)):
        for j in range(len(CAUSAL_LABELS)):
            ax.text(j, i, str(mat[i, j]),
                    ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black",
                    fontsize=10, fontweight="bold")

    ax.set_title("Causal Errors Per Combination\n(by true label)", fontsize=11, fontweight="bold")
    ax.set_xlabel("True Label", fontsize=10)
    ax.set_ylabel("Combination", fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + "heatmap_causal_errors.png", dpi=150)
    plt.close()
    print("  Saved: heatmap_causal_errors.png")

    # Relation error heatmap
    no_df = df[df[GT_CAUSAL] == "no"]
    present_labels = [l for l in RELATION_LABELS if l in no_df[GT_RELATION].values]

    relation_errors_matrix = []

    for combo in COMBINATIONS:
        row_errors = []
        for true_label in present_labels:
            subset = no_df[no_df[GT_RELATION] == true_label]
            errors = (
                subset[combo["relation"]].notna() &
                (subset[combo["relation"]] != true_label)
            ).sum()
            row_errors.append(errors)
        relation_errors_matrix.append(row_errors)

    fig, ax = plt.subplots(figsize=(10, 8))
    mat = np.array(relation_errors_matrix)
    im  = ax.imshow(mat, cmap="Reds", aspect="auto")
    plt.colorbar(im, ax=ax, label="Number of Errors")

    ax.set_xticks(range(len(present_labels)))
    ax.set_xticklabels(present_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(combo_names)))
    ax.set_yticklabels(combo_names, fontsize=9)

    for i in range(len(combo_names)):
        for j in range(len(present_labels)):
            ax.text(j, i, str(mat[i, j]),
                    ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black",
                    fontsize=9, fontweight="bold")

    ax.set_title("Relation Type Errors Per Combination\n(by true label)", fontsize=11, fontweight="bold")
    ax.set_xlabel("True Relation Type Label", fontsize=10)
    ax.set_ylabel("Combination", fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + "heatmap_relation_errors.png", dpi=150)
    plt.close()
    print("  Saved: heatmap_relation_errors.png")

    print()
    print("=" * 60)
    print("Script 4 complete. All outputs saved to results/")
    print("=" * 60)
    print()
    print("FULL PROJECT PIPELINE COMPLETE.")
    print("All 4 scripts have run successfully.")
    print("Check the results/ folder for all outputs.")
    print("=" * 60)


# ── RUN ──────────────────────────────────────────────────────

if __name__ == "__main__":
    run_error_analysis()
