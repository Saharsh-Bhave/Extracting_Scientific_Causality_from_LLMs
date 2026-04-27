# ============================================================
# 1_evaluate.py
#
# PURPOSE:
# This script calculates how well each LLM + prompt combination
# performed compared to the human annotations (ground truth).
#
# It evaluates two tasks separately:
#   Task 1 — Causal classification (yes / no)
#   Task 2 — Relation type classification (when causal = no)
#
# It produces scores:
#   - Overall (across all disciplines)
#   - Per discipline
#
# OUTPUT FILES (saved to results/ folder):
#   - scores_overall.csv
#   - scores_per_discipline.csv
#   - bar_chart_overall.png
#   - bar_chart_per_discipline.png
# ============================================================


# ── IMPORTS ─────────────────────────────────────────────────

import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter


# ── CONFIGURATION ────────────────────────────────────────────

DATA_PATH   = "data/dataset_clean.xlsx"
RESULTS_PATH = "results/"

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

def weighted_f1(y_true, y_pred):
    """
    Calculates Weighted F1 score manually without sklearn.

    For each unique label:
      - Precision = how many times the model was right when it predicted this label
      - Recall    = how many actual instances of this label the model found
      - F1        = balanced score between precision and recall

    Weighted F1 = average of each label's F1, weighted by how often that label appears.

    Returns a float between 0.0 and 1.0. Higher = better.
    """

    # Keep only pairs where both true and predicted values exist
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if pd.notna(t) and pd.notna(p)]

    # Need at least 2 valid pairs to calculate anything meaningful
    if len(pairs) < 2:
        return 0.0

    true_labels, pred_labels = zip(*pairs)

    # Get all unique labels that appear in the true labels
    unique_labels = set(true_labels)

    # Count how many times each true label appears
    # This is used to calculate the weight of each label
    label_counts = Counter(true_labels)
    total = len(true_labels)

    weighted_f1_score = 0.0

    for label in unique_labels:

        # True Positive: model predicted this label AND it was correct
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p == label)

        # False Positive: model predicted this label BUT it was wrong
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != label and p == label)

        # False Negative: actual label was this BUT model predicted something else
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p != label)

        # Precision: of all times model said this label, how often was it right?
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall: of all actual instances of this label, how many did model find?
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1: harmonic mean of precision and recall
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Weight = how often this label appears in the true labels
        weight = label_counts[label] / total

        # Add this label's weighted contribution to the total
        weighted_f1_score += weight * f1

    return weighted_f1_score


def evaluate_combo(df, combo):
    """
    Evaluates one LLM + prompt combination on a given dataframe.
    Returns F1 for causal task, F1 for relation task, and average.
    """

    # Task 1: Causal F1 (yes/no classification)
    f1_causal = weighted_f1(
        df[GT_CAUSAL],
        df[combo["causal"]]
    )

    # Task 2: Relation F1
    # Only evaluate on rows where manual_causal = "no"
    # Because relation type only applies to non-causal papers
    no_rows = df[df[GT_CAUSAL] == "no"]

    f1_relation = weighted_f1(
        no_rows[GT_RELATION],
        no_rows[combo["relation"]]
    )

    avg_f1 = round((f1_causal + f1_relation) / 2, 4)

    return round(f1_causal, 4), round(f1_relation, 4), avg_f1


# ── CHART FUNCTIONS ──────────────────────────────────────────

def plot_overall(overall_df):
    """Generates bar chart for overall scores."""

    # Colors: blue for GPT4, orange for Claude, green for Gemini
    colors = (
        ["#4C72B0"] * 3 +
        ["#DD8452"] * 3 +
        ["#55A868"] * 3
    )

    fig, ax = plt.subplots(figsize=(13, 6))

    bars = ax.bar(
        overall_df["Combination"],
        overall_df["Avg_F1"],
        color=colors,
        edgecolor="white",
        width=0.6
    )

    # Add score labels on top of each bar
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.005,
            f"{h:.4f}",
            ha="center", va="bottom", fontsize=9
        )

    # Add a legend to explain the colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="GPT-4"),
        Patch(facecolor="#DD8452", label="Claude"),
        Patch(facecolor="#55A868", label="Gemini"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_title("Overall Avg F1 Score — All 9 Combinations", fontsize=14, fontweight="bold")
    ax.set_xlabel("Combination", fontsize=11)
    ax.set_ylabel("Weighted F1 Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + "bar_chart_overall.png", dpi=150)
    plt.close()
    print("  Saved: bar_chart_overall.png")


def plot_per_discipline(discipline_df):
    """Generates 4 subplots — one per discipline."""

    colors = (
        ["#4C72B0"] * 3 +
        ["#DD8452"] * 3 +
        ["#55A868"] * 3
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, discipline in enumerate(DISCIPLINES):

        disc_data = discipline_df[discipline_df["Discipline"] == discipline]
        ax = axes[i]

        bars = ax.bar(
            disc_data["Combination"],
            disc_data["Avg_F1"],
            color=colors,
            edgecolor="white",
            width=0.6
        )

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.3f}",
                ha="center", va="bottom", fontsize=7
            )

        ax.set_title(discipline, fontsize=10, fontweight="bold")
        ax.set_ylabel("Avg F1", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="x", rotation=40, labelsize=7)

    plt.suptitle("Avg F1 Score Per Discipline — All 9 Combinations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + "bar_chart_per_discipline.png", dpi=150)
    plt.close()
    print("  Saved: bar_chart_per_discipline.png")


# ── MAIN ─────────────────────────────────────────────────────

def run_evaluation():

    # Load cleaned dataset
    print("Loading dataset...")
    df = pd.read_excel(DATA_PATH)
    print(f"Loaded {len(df)} rows | {df['discipline'].nunique()} disciplines")
    print()

    os.makedirs(RESULTS_PATH, exist_ok=True)


    # ── OVERALL ──────────────────────────────────────────────

    print("=" * 60)
    print("OVERALL EVALUATION (All Disciplines Combined)")
    print("=" * 60)

    overall_results = []

    for combo in COMBINATIONS:
        f1_c, f1_r, avg = evaluate_combo(df, combo)
        overall_results.append({
            "Combination" : combo["name"],
            "F1_Causal"   : f1_c,
            "F1_Relation" : f1_r,
            "Avg_F1"      : avg
        })
        print(f"  {combo['name']:<24} | Causal F1: {f1_c:.4f} | Relation F1: {f1_r:.4f} | Avg: {avg:.4f}")

    overall_df = pd.DataFrame(overall_results)
    overall_df = overall_df.sort_values("Avg_F1", ascending=False).reset_index(drop=True)
    overall_df.insert(0, "Rank", range(1, len(overall_df) + 1))

    overall_df.to_csv(RESULTS_PATH + "scores_overall.csv", index=False)
    print(f"\n  Saved: scores_overall.csv")


    # ── PER DISCIPLINE ───────────────────────────────────────

    print()
    print("=" * 60)
    print("PER DISCIPLINE EVALUATION")
    print("=" * 60)

    discipline_results = []

    for discipline in DISCIPLINES:
        disc_df = df[df["discipline"] == discipline]
        print(f"\n  --- {discipline} ({len(disc_df)} papers) ---")

        for combo in COMBINATIONS:
            f1_c, f1_r, avg = evaluate_combo(disc_df, combo)
            discipline_results.append({
                "Discipline"  : discipline,
                "Combination" : combo["name"],
                "F1_Causal"   : f1_c,
                "F1_Relation" : f1_r,
                "Avg_F1"      : avg
            })
            print(f"    {combo['name']:<24} | Causal: {f1_c:.4f} | Relation: {f1_r:.4f} | Avg: {avg:.4f}")

    discipline_df = pd.DataFrame(discipline_results)
    discipline_df = discipline_df.sort_values(
        ["Discipline", "Avg_F1"], ascending=[True, False]
    ).reset_index(drop=True)

    discipline_df.to_csv(RESULTS_PATH + "scores_per_discipline.csv", index=False)
    print(f"\n  Saved: scores_per_discipline.csv")


    # ── SUMMARY ──────────────────────────────────────────────

    print()
    print("=" * 60)
    print("SUMMARY — WINNERS")
    print("=" * 60)

    best_overall = overall_df.iloc[0]
    print(f"\n  Best Overall         : {best_overall['Combination']} (Avg F1: {best_overall['Avg_F1']})")

    for discipline in DISCIPLINES:
        best = discipline_df[discipline_df["Discipline"] == discipline].iloc[0]
        print(f"  Best in {discipline:<36}: {best['Combination']} (Avg F1: {best['Avg_F1']})")

    # Best LLM — average across its 3 prompt strategies
    overall_df["LLM"]    = overall_df["Combination"].str.split(" ").str[0]
    overall_df["Prompt"] = overall_df["Combination"].str.split(" ", n=1).str[1]

    best_llm    = overall_df.groupby("LLM")["Avg_F1"].mean().idxmax()
    best_prompt = overall_df.groupby("Prompt")["Avg_F1"].mean().idxmax()

    llm_score    = overall_df.groupby("LLM")["Avg_F1"].mean().max()
    prompt_score = overall_df.groupby("Prompt")["Avg_F1"].mean().max()

    print(f"\n  Best LLM             : {best_llm} (Mean Avg F1: {llm_score:.4f})")
    print(f"  Best Prompt Strategy : {best_prompt} (Mean Avg F1: {prompt_score:.4f})")


    # ── CHARTS ───────────────────────────────────────────────

    print()
    print("Generating charts...")
    plot_overall(overall_df)
    plot_per_discipline(discipline_df)

    print()
    print("=" * 60)
    print("Script 1 complete. All outputs saved to results/")
    print("=" * 60)


# ── RUN ──────────────────────────────────────────────────────

if __name__ == "__main__":
    run_evaluation()
