# ============================================================
# 2_agreement.py
#
# PURPOSE:
# This script measures how much the 3 LLMs agreed with each
# other on every paper — independently of whether they were
# right or wrong.
#
# For each paper, all 9 combinations vote on:
#   - Causal: yes or no
#   - Relation type (if causal = no)
#
# Each paper is then tagged into one of 3 categories:
#   EASY              → all or most combinations agreed AND were correct
#   AMBIGUOUS         → combinations disagreed with each other
#   SYSTEMATIC FAILURE → all combinations agreed BUT were wrong
#
# OUTPUT FILES (saved to results/ folder):
#   - agreement_table.csv      → every paper tagged with its category
#   - agreement_summary.csv    → count of each category per discipline
#   - agreement_pie_chart.png  → pie chart of Easy/Ambiguous/Systematic
#   - agreement_bar_chart.png  → breakdown per discipline
# ============================================================


# ── IMPORTS ─────────────────────────────────────────────────

import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter


# ── CONFIGURATION ────────────────────────────────────────────

DATA_PATH    = "data/dataset_clean.xlsx"
RESULTS_PATH = "results/"

GT_CAUSAL   = "manual_causal"
GT_RELATION = "manual_relation_type"

DISCIPLINES = [
    "Biomedical Engineering",
    "Molecular Biology / Genetics",
    "Cancer Biology",
    "Chemistry"
]

# All 9 LLM + prompt combinations
# Same as Script 1 — consistent naming throughout
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

# Agreement threshold — how many out of 9 must agree to call it "agreed"
# 7 out of 9 means at least 77% of combinations said the same thing
AGREEMENT_THRESHOLD = 7


# ── HELPER FUNCTIONS ─────────────────────────────────────────

def get_majority_vote(votes):
    """
    Given a list of predictions, returns the value that appears most often.
    If there is a tie, returns the first one found.

    Example:
        votes = ['yes', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'yes']
        returns 'yes' (appears 6 times vs 3)
    """
    # Filter out NaN values — only count actual predictions
    valid_votes = [v for v in votes if pd.notna(v)]

    # If no valid votes exist, return None
    if not valid_votes:
        return None

    # Counter counts how many times each value appears
    # .most_common(1) returns the single most frequent value
    return Counter(valid_votes).most_common(1)[0][0]


def get_agreement_count(votes):
    """
    Returns how many combinations agreed on the majority vote.

    Example:
        votes = ['yes', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes']
        majority = 'yes' (appears 7 times)
        returns 7
    """
    valid_votes = [v for v in votes if pd.notna(v)]
    if not valid_votes:
        return 0

    majority = Counter(valid_votes).most_common(1)[0][0]
    return sum(1 for v in valid_votes if v == majority)


def categorize_paper(majority_vote, ground_truth, agreement_count, total_combos=9):
    """
    Categorizes a paper into one of 3 categories based on:
      - How many combinations agreed (agreement_count)
      - Whether the majority agreed answer matches ground truth

    Categories:
      EASY              → high agreement AND correct
      AMBIGUOUS         → low agreement (models disagreed)
      SYSTEMATIC FAILURE → high agreement BUT wrong
    """

    # If majority vote or ground truth is missing, skip
    if majority_vote is None or pd.isna(ground_truth):
        return "Unknown"

    agreed = agreement_count >= AGREEMENT_THRESHOLD
    correct = (majority_vote == ground_truth)

    if agreed and correct:
        return "Easy"
    elif not agreed:
        return "Ambiguous"
    else:
        # agreed but NOT correct
        return "Systematic Failure"


# ── MAIN FUNCTION ────────────────────────────────────────────

def run_agreement():

    # Load cleaned dataset
    print("Loading dataset...")
    df = pd.read_excel(DATA_PATH)
    print(f"Loaded {len(df)} rows | {df['discipline'].nunique()} disciplines")
    print()

    os.makedirs(RESULTS_PATH, exist_ok=True)

    # ── PER PAPER ANALYSIS ───────────────────────────────────
    # For every paper, calculate agreement across all 9 combinations

    print("=" * 60)
    print("CALCULATING AGREEMENT PER PAPER")
    print("=" * 60)

    paper_results = []
    # paper_results will hold one row per paper with all agreement data

    for idx, row in df.iterrows():
        # idx = row number, row = the full data for that paper

        # ── CAUSAL AGREEMENT ─────────────────────────────────

        # Collect all 9 causal predictions for this paper
        causal_votes = [row[combo["causal"]] for combo in COMBINATIONS]

        # Find majority vote across 9 combinations
        causal_majority = get_majority_vote(causal_votes)

        # Count how many combinations agreed on that majority
        causal_agreement_count = get_agreement_count(causal_votes)

        # Ground truth causal value for this paper
        gt_causal = row[GT_CAUSAL]

        # Categorize this paper based on causal agreement
        causal_category = categorize_paper(
            causal_majority,
            gt_causal,
            causal_agreement_count
        )

        # ── RELATION AGREEMENT ────────────────────────────────
        # Only applies to papers where manual_causal = "no"

        relation_majority  = None
        relation_agreement = 0
        relation_category  = "N/A"
        # Default to N/A — will be filled in only for "no" papers

        if pd.notna(gt_causal) and gt_causal == "no":

            # Collect all 9 relation predictions for this paper
            relation_votes = [row[combo["relation"]] for combo in COMBINATIONS]

            relation_majority  = get_majority_vote(relation_votes)
            relation_agreement = get_agreement_count(relation_votes)
            gt_relation        = row[GT_RELATION]

            relation_category = categorize_paper(
                relation_majority,
                gt_relation,
                relation_agreement
            )

        # ── STORE RESULTS FOR THIS PAPER ─────────────────────

        paper_results.append({
            "num"                    : row["num"],
            "title"                  : row["title"],
            "discipline"             : row["discipline"],
            "manual_causal"          : gt_causal,
            "manual_relation_type"   : row[GT_RELATION],

            # Causal agreement data
            "causal_majority_vote"   : causal_majority,
            "causal_agreement_count" : causal_agreement_count,
            "causal_category"        : causal_category,

            # Relation agreement data
            "relation_majority_vote" : relation_majority,
            "relation_agreement_count": relation_agreement,
            "relation_category"      : relation_category,
        })

    # Convert to DataFrame
    agreement_df = pd.DataFrame(paper_results)

    # Save full paper-level agreement table
    agreement_df.to_csv(RESULTS_PATH + "agreement_table.csv", index=False)
    print(f"  Saved: agreement_table.csv ({len(agreement_df)} papers)")


    # ── SUMMARY BY DISCIPLINE ────────────────────────────────
    # Count how many papers fall into each category per discipline

    print()
    print("=" * 60)
    print("AGREEMENT SUMMARY PER DISCIPLINE")
    print("=" * 60)

    summary_results = []

    for discipline in DISCIPLINES:

        disc_data = agreement_df[agreement_df["discipline"] == discipline]

        # Count causal categories for this discipline
        causal_counts = disc_data["causal_category"].value_counts()

        # Count relation categories (only "no" papers)
        no_papers = disc_data[disc_data["manual_causal"] == "no"]
        relation_counts = no_papers["relation_category"].value_counts()

        total = len(disc_data)
        total_no = len(no_papers)

        print(f"\n  --- {discipline} ({total} papers, {total_no} non-causal) ---")

        print(f"  CAUSAL AGREEMENT:")
        for cat in ["Easy", "Ambiguous", "Systematic Failure"]:
            count = causal_counts.get(cat, 0)
            pct   = round(count / total * 100, 1) if total > 0 else 0
            print(f"    {cat:<22}: {count} papers ({pct}%)")

        print(f"  RELATION AGREEMENT (non-causal papers only):")
        for cat in ["Easy", "Ambiguous", "Systematic Failure"]:
            count = relation_counts.get(cat, 0)
            pct   = round(count / total_no * 100, 1) if total_no > 0 else 0
            print(f"    {cat:<22}: {count} papers ({pct}%)")

        summary_results.append({
            "Discipline"                      : discipline,
            "Total_Papers"                    : total,
            "Causal_Easy"                     : causal_counts.get("Easy", 0),
            "Causal_Ambiguous"                : causal_counts.get("Ambiguous", 0),
            "Causal_Systematic_Failure"       : causal_counts.get("Systematic Failure", 0),
            "NonCausal_Papers"                : total_no,
            "Relation_Easy"                   : relation_counts.get("Easy", 0),
            "Relation_Ambiguous"              : relation_counts.get("Ambiguous", 0),
            "Relation_Systematic_Failure"     : relation_counts.get("Systematic Failure", 0),
        })

    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(RESULTS_PATH + "agreement_summary.csv", index=False)
    print(f"\n  Saved: agreement_summary.csv")


    # ── OVERALL AGREEMENT STATS ──────────────────────────────

    print()
    print("=" * 60)
    print("OVERALL AGREEMENT STATISTICS")
    print("=" * 60)

    total_papers = len(agreement_df)

    for cat in ["Easy", "Ambiguous", "Systematic Failure"]:
        count = (agreement_df["causal_category"] == cat).sum()
        pct   = round(count / total_papers * 100, 1)
        print(f"  Causal {cat:<22}: {count} / {total_papers} papers ({pct}%)")

    print()
    no_papers_all = agreement_df[agreement_df["manual_causal"] == "no"]
    total_no = len(no_papers_all)

    for cat in ["Easy", "Ambiguous", "Systematic Failure"]:
        count = (no_papers_all["relation_category"] == cat).sum()
        pct   = round(count / total_no * 100, 1) if total_no > 0 else 0
        print(f"  Relation {cat:<22}: {count} / {total_no} papers ({pct}%)")

    # Average agreement count across all papers
    avg_causal_agreement = agreement_df["causal_agreement_count"].mean()
    print(f"\n  Avg causal agreement count : {avg_causal_agreement:.2f} / 9 combinations")


    # ── CHARTS ───────────────────────────────────────────────

    print()
    print("Generating charts...")

    # Chart 1 — Overall Pie Chart (Causal Agreement Categories)
    causal_cats = agreement_df["causal_category"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left pie: causal agreement
    axes[0].pie(
        causal_cats.values,
        labels=causal_cats.index,
        autopct="%1.1f%%",       # show percentage on each slice
        colors=["#55A868", "#DD8452", "#C44E52"],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    axes[0].set_title("Causal Agreement\n(All Papers)", fontsize=12, fontweight="bold")

    # Right pie: relation agreement (non-causal papers only)
    no_data = agreement_df[agreement_df["manual_causal"] == "no"]
    relation_cats = no_data["relation_category"].value_counts()

    axes[1].pie(
        relation_cats.values,
        labels=relation_cats.index,
        autopct="%1.1f%%",
        colors=["#55A868", "#DD8452", "#C44E52"],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    axes[1].set_title("Relation Type Agreement\n(Non-Causal Papers Only)", fontsize=12, fontweight="bold")

    plt.suptitle("Inter-LLM Agreement Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + "agreement_pie_chart.png", dpi=150)
    plt.close()
    print("  Saved: agreement_pie_chart.png")


    # Chart 2 — Stacked Bar Chart Per Discipline
    categories  = ["Easy", "Ambiguous", "Systematic Failure"]
    bar_colors  = ["#55A868", "#DD8452", "#C44E52"]
    disc_labels = [d.replace(" / ", "/\n") for d in DISCIPLINES]
    # Shorten labels so they fit on the chart

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, task in enumerate(["Causal", "Relation"]):

        ax = axes[ax_idx]
        bottom = [0] * len(DISCIPLINES)
        # bottom tracks where each bar segment starts stacking from

        for cat, color in zip(categories, bar_colors):

            values = []
            for disc in DISCIPLINES:
                row = summary_df[summary_df["Discipline"] == disc].iloc[0]

                if task == "Causal":
                    col  = f"Causal_{cat.replace(' ', '_')}"
                    total = row["Total_Papers"]
                else:
                    col  = f"Relation_{cat.replace(' ', '_')}"
                    total = row["NonCausal_Papers"]

                # Convert count to percentage for fair comparison
                count = row[col]
                pct   = (count / total * 100) if total > 0 else 0
                values.append(pct)

            bars = ax.bar(
                disc_labels,
                values,
                bottom=bottom,
                label=cat,
                color=color,
                edgecolor="white",
                width=0.5
            )

            # Update bottom for next stack layer
            bottom = [b + v for b, v in zip(bottom, values)]

        ax.set_title(f"{task} Agreement by Discipline", fontsize=11, fontweight="bold")
        ax.set_ylabel("Percentage of Papers (%)", fontsize=9)
        ax.set_ylim(0, 110)
        ax.legend(loc="upper right", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)

    plt.suptitle("Agreement Categories Per Discipline", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + "agreement_bar_chart.png", dpi=150)
    plt.close()
    print("  Saved: agreement_bar_chart.png")

    print()
    print("=" * 60)
    print("Script 2 complete. All outputs saved to results/")
    print("=" * 60)

    # Return agreement_df so Script 3 can use it directly
    return agreement_df


# ── RUN ──────────────────────────────────────────────────────

if __name__ == "__main__":
    run_agreement()
