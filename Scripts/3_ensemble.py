# ============================================================
# 3_ensemble.py
#
# PURPOSE:
# This script builds an ensemble model using majority voting.
# Instead of relying on any single LLM + prompt combination,
# all 9 combinations vote on each paper and the majority
# answer becomes the final ensemble prediction.
#
# It then evaluates the ensemble the same way Script 1
# evaluated individual combinations — using Weighted F1.
#
# Finally it compares the ensemble score against all 9
# individual combination scores to answer:
#   "Does combining all models beat any single one?"
#
# INPUT FILES (from previous scripts):
#   - data/dataset_clean.xlsx      (from Script 0)
#   - results/scores_overall.csv   (from Script 1)
#   - results/agreement_table.csv  (from Script 2)
#
# OUTPUT FILES (saved to results/ folder):
#   - ensemble_predictions.csv     → majority vote per paper
#   - ensemble_results.csv         → ensemble F1 vs all 9 combos
#   - ensemble_vs_individual.png   → bar chart comparison
#   - ensemble_by_discipline.png   → ensemble score per discipline
# ============================================================


# ── IMPORTS ─────────────────────────────────────────────────

import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.patches import Patch


# ── CONFIGURATION ────────────────────────────────────────────

DATA_PATH        = "data/dataset_clean.xlsx"
SCORES_PATH      = "results/scores_overall.csv"
AGREEMENT_PATH   = "results/agreement_table.csv"
RESULTS_PATH     = "results/"

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


# ── HELPER FUNCTIONS ─────────────────────────────────────────

def majority_vote(votes):
    """
    Returns the most common value from a list of predictions.
    Ignores NaN values.
    If there is a tie, returns the first most common value found.

    Example:
        votes = ['yes','yes','no','yes','no','yes','no','yes','yes']
        returns 'yes'  (appears 6 times)
    """
    valid = [v for v in votes if pd.notna(v)]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def vote_breakdown(votes):
    """
    Returns a readable string showing how votes were split.

    Example:
        votes = ['yes','yes','no','yes','no','yes','no','yes','yes']
        returns 'yes:6 | no:3'
    """
    valid = [v for v in votes if pd.notna(v)]
    if not valid:
        return "no votes"
    counts = Counter(valid)
    return " | ".join([f"{k}:{v}" for k, v in counts.most_common()])


def weighted_f1(y_true, y_pred):
    """
    Calculates Weighted F1 score manually.
    Same function used in Script 1 for consistency.

    For each label:
      Precision = TP / (TP + FP)
      Recall    = TP / (TP + FN)
      F1        = 2 * P * R / (P + R)

    Weighted F1 = sum of (weight * F1) for each label
    Weight = how often that label appears in ground truth
    """
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if pd.notna(t) and pd.notna(p)]
    if len(pairs) < 2:
        return 0.0

    true_labels, pred_labels = zip(*pairs)
    unique_labels = set(true_labels)
    label_counts  = Counter(true_labels)
    total         = len(true_labels)
    score         = 0.0

    for label in unique_labels:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p == label)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != label and p == label)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        weight    = label_counts[label] / total
        score    += weight * f1

    return score


# ── MAIN FUNCTION ────────────────────────────────────────────

def run_ensemble():

    # Load datasets
    print("Loading datasets...")
    df          = pd.read_excel(DATA_PATH)
    scores_df   = pd.read_csv(SCORES_PATH)
    agreement_df = pd.read_csv(AGREEMENT_PATH)

    print(f"  Dataset       : {len(df)} rows")
    print(f"  Scores table  : {len(scores_df)} combinations")
    print(f"  Agreement table: {len(agreement_df)} papers")
    print()

    os.makedirs(RESULTS_PATH, exist_ok=True)


    # ── STEP 1: BUILD ENSEMBLE PREDICTIONS ───────────────────
    # For every paper, collect all 9 votes and pick the majority

    print("=" * 60)
    print("BUILDING ENSEMBLE PREDICTIONS")
    print("=" * 60)

    predictions = []
    # predictions holds one row per paper with ensemble results

    for idx, row in df.iterrows():

        # Collect all 9 causal votes for this paper
        causal_votes = [row[combo["causal"]] for combo in COMBINATIONS]

        # Majority vote for causal
        ensemble_causal = majority_vote(causal_votes)

        # Vote breakdown string for transparency
        causal_breakdown = vote_breakdown(causal_votes)

        # Collect all 9 relation votes for this paper
        relation_votes = [row[combo["relation"]] for combo in COMBINATIONS]

        # Majority vote for relation type
        ensemble_relation = majority_vote(relation_votes)

        # Vote breakdown string
        relation_breakdown = vote_breakdown(relation_votes)

        # Was the ensemble causal prediction correct?
        causal_correct = (
            ensemble_causal == row[GT_CAUSAL]
            if pd.notna(row[GT_CAUSAL]) else None
        )

        # Was the ensemble relation prediction correct?
        # Only relevant for non-causal papers
        relation_correct = None
        if pd.notna(row[GT_CAUSAL]) and row[GT_CAUSAL] == "no":
            relation_correct = (
                ensemble_relation == row[GT_RELATION]
                if pd.notna(row[GT_RELATION]) else None
            )

        predictions.append({
            "num"                  : row["num"],
            "title"                : row["title"],
            "discipline"           : row["discipline"],
            "manual_causal"        : row[GT_CAUSAL],
            "manual_relation_type" : row[GT_RELATION],
            "ensemble_causal"      : ensemble_causal,
            "ensemble_relation"    : ensemble_relation,
            "causal_vote_breakdown": causal_breakdown,
            "relation_vote_breakdown": relation_breakdown,
            "causal_correct"       : causal_correct,
            "relation_correct"     : relation_correct,
        })

    pred_df = pd.DataFrame(predictions)

    # Save ensemble predictions
    pred_df.to_csv(RESULTS_PATH + "ensemble_predictions.csv", index=False)
    print(f"  Saved: ensemble_predictions.csv ({len(pred_df)} papers)")

    # Quick accuracy check
    causal_acc = pred_df["causal_correct"].sum() / pred_df["causal_correct"].notna().sum()
    print(f"  Ensemble Causal Accuracy  : {causal_acc:.4f}")

    no_papers = pred_df[pred_df["manual_causal"] == "no"]
    if len(no_papers) > 0:
        rel_acc = no_papers["relation_correct"].sum() / no_papers["relation_correct"].notna().sum()
        print(f"  Ensemble Relation Accuracy: {rel_acc:.4f}")


    # ── STEP 2: EVALUATE ENSEMBLE WITH WEIGHTED F1 ───────────
    # Calculate ensemble F1 same way Script 1 calculated individual F1

    print()
    print("=" * 60)
    print("ENSEMBLE WEIGHTED F1 EVALUATION")
    print("=" * 60)

    # Overall ensemble F1
    f1_causal_overall = weighted_f1(
        pred_df[GT_CAUSAL],
        pred_df["ensemble_causal"]
    )

    no_df = pred_df[pred_df[GT_CAUSAL] == "no"]
    f1_relation_overall = weighted_f1(
        no_df[GT_RELATION],
        no_df["ensemble_relation"]
    )

    avg_f1_overall = round((f1_causal_overall + f1_relation_overall) / 2, 4)

    print(f"\n  OVERALL ENSEMBLE:")
    print(f"    Causal F1   : {f1_causal_overall:.4f}")
    print(f"    Relation F1 : {f1_relation_overall:.4f}")
    print(f"    Avg F1      : {avg_f1_overall:.4f}")

    # Per discipline ensemble F1
    print(f"\n  PER DISCIPLINE:")

    discipline_ensemble = []

    for discipline in DISCIPLINES:
        disc = pred_df[pred_df["discipline"] == discipline]
        disc_no = disc[disc[GT_CAUSAL] == "no"]

        f1_c = weighted_f1(disc[GT_CAUSAL], disc["ensemble_causal"])
        f1_r = weighted_f1(disc_no[GT_RELATION], disc_no["ensemble_relation"])
        avg  = round((f1_c + f1_r) / 2, 4)

        print(f"    {discipline:<38}: Causal {f1_c:.4f} | Relation {f1_r:.4f} | Avg {avg:.4f}")

        discipline_ensemble.append({
            "Discipline"  : discipline,
            "F1_Causal"   : round(f1_c, 4),
            "F1_Relation" : round(f1_r, 4),
            "Avg_F1"      : avg
        })


    # ── STEP 3: COMPARE ENSEMBLE VS INDIVIDUAL COMBOS ────────
    # Build a final comparison table

    print()
    print("=" * 60)
    print("ENSEMBLE vs INDIVIDUAL COMBINATIONS")
    print("=" * 60)

    # Load individual scores from Script 1 output
    comparison = scores_df[["Rank", "Combination", "F1_Causal", "F1_Relation", "Avg_F1"]].copy()

    # Add ensemble as a new row
    ensemble_row = pd.DataFrame([{
        "Rank"        : "–",
        "Combination" : "⭐ Ensemble (Majority Vote)",
        "F1_Causal"   : round(f1_causal_overall, 4),
        "F1_Relation" : round(f1_relation_overall, 4),
        "Avg_F1"      : avg_f1_overall
    }])

    comparison = pd.concat([ensemble_row, comparison], ignore_index=True)

    # Sort by Avg_F1 descending to see where ensemble ranks
    comparison_sorted = comparison.copy()
    comparison_sorted["sort_key"] = pd.to_numeric(comparison_sorted["Avg_F1"], errors="coerce")
    comparison_sorted = comparison_sorted.sort_values("sort_key", ascending=False).drop("sort_key", axis=1)

    print()
    print(comparison_sorted.to_string(index=False))

    # Check if ensemble beats the best individual combination
    best_individual_f1 = scores_df["Avg_F1"].max()
    best_individual    = scores_df.loc[scores_df["Avg_F1"].idxmax(), "Combination"]

    print()
    if avg_f1_overall > best_individual_f1:
        gap = avg_f1_overall - best_individual_f1
        print(f"  ✅ Ensemble OUTPERFORMS the best individual combination!")
        print(f"     Best Individual : {best_individual} (Avg F1: {best_individual_f1:.4f})")
        print(f"     Ensemble        : Avg F1 {avg_f1_overall:.4f} (+{gap:.4f})")
    else:
        gap = best_individual_f1 - avg_f1_overall
        print(f"  ℹ️  Ensemble does NOT outperform the best individual combination.")
        print(f"     Best Individual : {best_individual} (Avg F1: {best_individual_f1:.4f})")
        print(f"     Ensemble        : Avg F1 {avg_f1_overall:.4f} (-{gap:.4f})")
        print(f"     Finding: Models share correlated errors — they fail on the same papers.")

    # Save full comparison table
    comparison_sorted.to_csv(RESULTS_PATH + "ensemble_results.csv", index=False)
    print(f"\n  Saved: ensemble_results.csv")


    # ── STEP 4: CHARTS ───────────────────────────────────────

    print()
    print("Generating charts...")

    # Chart 1 — Ensemble vs All Individual Combinations (Bar Chart)
    plot_df = comparison_sorted.copy()
    plot_df["Avg_F1"] = pd.to_numeric(plot_df["Avg_F1"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Avg_F1"])

    # Color the ensemble bar differently to make it stand out
    bar_colors = []
    for name in plot_df["Combination"]:
        if "Ensemble" in str(name):
            bar_colors.append("#C44E52")   # red = ensemble
        elif "GPT4" in str(name):
            bar_colors.append("#4C72B0")   # blue = GPT4
        elif "Claude" in str(name):
            bar_colors.append("#DD8452")   # orange = Claude
        else:
            bar_colors.append("#55A868")   # green = Gemini

    fig, ax = plt.subplots(figsize=(14, 6))

    bars = ax.bar(
        plot_df["Combination"],
        plot_df["Avg_F1"],
        color=bar_colors,
        edgecolor="white",
        width=0.6
    )

    # Add value labels on top of each bar
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.005,
            f"{h:.4f}",
            ha="center", va="bottom", fontsize=8
        )

    # Legend
    legend_elements = [
        Patch(facecolor="#C44E52", label="Ensemble (Majority Vote)"),
        Patch(facecolor="#4C72B0", label="GPT-4"),
        Patch(facecolor="#DD8452", label="Claude"),
        Patch(facecolor="#55A868", label="Gemini"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_title("Ensemble vs Individual Combinations — Avg F1 Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("Combination", fontsize=11)
    ax.set_ylabel("Avg Weighted F1 Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis="x", rotation=38)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + "ensemble_vs_individual.png", dpi=150)
    plt.close()
    print("  Saved: ensemble_vs_individual.png")


    # Chart 2 — Ensemble Score Per Discipline
    disc_ens_df = pd.DataFrame(discipline_ensemble)

    fig, ax = plt.subplots(figsize=(10, 5))

    x      = range(len(DISCIPLINES))
    width  = 0.25
    labels = [d.replace(" / ", "/\n") for d in DISCIPLINES]

    bars1 = ax.bar(
        [i - width for i in x],
        disc_ens_df["F1_Causal"],
        width=width,
        label="Causal F1",
        color="#4C72B0",
        edgecolor="white"
    )
    bars2 = ax.bar(
        x,
        disc_ens_df["F1_Relation"],
        width=width,
        label="Relation F1",
        color="#DD8452",
        edgecolor="white"
    )
    bars3 = ax.bar(
        [i + width for i in x],
        disc_ens_df["Avg_F1"],
        width=width,
        label="Avg F1",
        color="#C44E52",
        edgecolor="white"
    )

    ax.set_title("Ensemble Performance Per Discipline", fontsize=13, fontweight="bold")
    ax.set_ylabel("Weighted F1 Score", fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + "ensemble_by_discipline.png", dpi=150)
    plt.close()
    print("  Saved: ensemble_by_discipline.png")

    print()
    print("=" * 60)
    print("Script 3 complete. All outputs saved to results/")
    print("=" * 60)


# ── RUN ──────────────────────────────────────────────────────

if __name__ == "__main__":
    run_ensemble()
