"""
statistical_analysis.py
=======================
Script 5 of 5 — Statistical association: hotspot membership vs pathogenicity

PURPOSE:
    Build the 2×2 contingency table (hotspot membership × pathogenicity label)
    and test whether pathogenic ClinVar variants are significantly enriched at
    FoldX-predicted structural hotspot positions.

    Tests performed:
        • Fisher's exact test (scipy.stats.fisher_exact)
          — used when any expected cell count < 5
        • Chi-square test (scipy.stats.chi2_contingency)
          — used when all expected cell counts ≥ 5 (reported alongside Fisher)
        • Odds ratio with 95% confidence interval
          (Woolf/logit method: OR = (a*d)/(b*c), CI via log(OR) ± 1.96*SE)

    Multiple-testing note:
        Only one primary hypothesis is tested (hotspot enrichment), so no
        multiple-testing correction is required here.

INPUT:
    derived/variants_with_hotspot.csv  (output of map_variants.py)
    derived/contingency_counts.csv     (output of map_variants.py)

OUTPUT:
    derived/statistical_results.txt    — full results as a text report
    derived/contingency_table.csv      — formatted 2×2 table
    derived/contingency_heatmap.png    — heatmap of observed counts
    derived/mosaic_plot.png            — mosaic plot

USAGE:
    python statistical_analysis.py
"""

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET SUBSET — 866 VARIANTS (DOMAIN-RESTRICTED, residues 96–289)
# ═══════════════════════════════════════════════════════════════════════════════
# The full ClinVar labeled dataset contains 1,374 variants (874 pathogenic,
# 500 benign). This script restricts the Fisher test to the 866 variants
# (611 pathogenic, 255 benign) that map to positions 96–289, the region
# covered by the FoldX PositionScan.
#
# WHY THE RESTRICTION IS REQUIRED:
#   Variants outside residues 96–289 were never scanned by FoldX, so they
#   cannot be assigned a hotspot flag from the scan. Including them forces
#   is_hotspot=0 for all of them, artificially inflating the non-hotspot
#   column of the contingency table. This biases the Fisher test toward
#   a null result by diluting the true hotspot/non-hotspot signal.
#   The domain restriction is therefore mandatory for a valid test.
#
# The 1,374-variant full set is used in tp53_improved_classification.py,
# which extends structural coverage to outside-domain positions via the
# AlphaFold2 model (AF-P04637). The 437-variant subset used in
# compare_tools.py is the further intersection with available AlphaMissense
# scores.
# ═══════════════════════════════════════════════════════════════════════════════

import os
import math
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VARIANTS_FILE    = os.path.join("derived", "variants_with_hotspot.csv")
CONTINGENCY_FILE = os.path.join("derived", "contingency_counts.csv")
OUTPUT_DIR       = "derived"
ALPHA            = 0.05   # significance level


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def odds_ratio_ci(a: int, b: int, c: int, d: int, alpha: float = 0.05):
    """
    Compute odds ratio and (1-alpha)×100% CI via Woolf's logit method.
        Table layout:
                    hotspot   non-hotspot
        pathogenic    a           b
        benign        c           d

    Returns (OR, CI_lower, CI_upper).
    Adds 0.5 continuity correction if any cell is 0.
    """
    a_, b_, c_, d_ = float(a), float(b), float(c), float(d)
    if 0 in (a_, b_, c_, d_):
        a_ += 0.5; b_ += 0.5; c_ += 0.5; d_ += 0.5  # Haldane-Anscombe

    or_val = (a_ * d_) / (b_ * c_)
    log_or = math.log(or_val)
    se_log = math.sqrt(1/a_ + 1/b_ + 1/c_ + 1/d_)
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lo = math.exp(log_or - z * se_log)
    ci_hi = math.exp(log_or + z * se_log)
    return round(or_val, 4), round(ci_lo, 4), round(ci_hi, 4)


def expected_counts(table: np.ndarray) -> np.ndarray:
    """
    Compute expected cell counts for a 2×2 table:
        E_ij = (row_i_sum × col_j_sum) / grand_total
    """
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    return (row_sums * col_sums) / table.sum()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Statistical Analysis: Hotspot Enrichment")
    print("=" * 60)

    # 1. Load data
    if not os.path.exists(VARIANTS_FILE):
        print(f"[ERROR] Not found: {VARIANTS_FILE}")
        print("Run map_variants.py first.")
        return

    variants = pd.read_csv(VARIANTS_FILE)
    print(f"\nVariants loaded: {len(variants)}")

    # 2. Restrict to structural domain (residues 96–289) before building the table.
    #    Variants outside the domain are all forced to is_hotspot=0 because the
    #    scan never covered those positions — including them inflates the
    #    non-hotspot column and confounds the test.
    domain_variants = variants[
        variants["position"].between(96, 289)
    ].copy()
    n_outside = len(variants) - len(domain_variants)
    print(f"\nVariants inside structural domain (96-289): {len(domain_variants)}")
    print(f"Variants outside domain (excluded from test): {n_outside}")

    a = int(((domain_variants["label"] == "pathogenic") & (domain_variants["is_hotspot"] == 1)).sum())
    b = int(((domain_variants["label"] == "pathogenic") & (domain_variants["is_hotspot"] == 0)).sum())
    c = int(((domain_variants["label"] == "benign")     & (domain_variants["is_hotspot"] == 1)).sum())
    d = int(((domain_variants["label"] == "benign")     & (domain_variants["is_hotspot"] == 0)).sum())

    table = np.array([[a, b],
                      [c, d]])

    print("\n--- Observed 2×2 contingency table ---")
    print(f"{'':20s} {'Hotspot':>10} {'Non-hotspot':>13} {'Total':>8}")
    print(f"{'Pathogenic':20s} {a:>10} {b:>13} {a+b:>8}")
    print(f"{'Benign':20s} {c:>10} {d:>13} {c+d:>8}")
    print(f"{'Total':20s} {a+c:>10} {b+d:>13} {a+b+c+d:>8}")

    # 3. Expected counts
    exp = expected_counts(table)
    print("\n--- Expected cell counts (under H₀) ---")
    print(f"{'':20s} {'Hotspot':>10} {'Non-hotspot':>13}")
    print(f"{'Pathogenic':20s} {exp[0,0]:>10.2f} {exp[0,1]:>13.2f}")
    print(f"{'Benign':20s} {exp[1,0]:>10.2f} {exp[1,1]:>13.2f}")

    use_chisq = (exp >= 5).all()
    print(f"\n  All expected counts ≥ 5: {use_chisq}")
    print(f"  → Using {'chi-square + ' if use_chisq else ''}Fisher's exact test")

    # 4. Fisher's exact test
    odds_ratio_fisher, p_fisher = stats.fisher_exact(table, alternative="two-sided")
    print(f"\n--- Fisher's exact test ---")
    print(f"  Odds ratio (Fisher) : {odds_ratio_fisher:.4f}")
    print(f"  p-value             : {p_fisher:.6e}")
    sig = "SIGNIFICANT" if p_fisher < ALPHA else "not significant"
    print(f"  Result (α={ALPHA})   : {sig}")

    # 5. Woolf OR with 95% CI
    or_woolf, ci_lo, ci_hi = odds_ratio_ci(a, b, c, d)
    print(f"\n--- Odds ratio (Woolf/logit method) ---")
    print(f"  OR          : {or_woolf:.4f}")
    print(f"  95% CI      : ({ci_lo:.4f}, {ci_hi:.4f})")
    if ci_lo > 1:
        print("  Interpretation: CI excludes 1 → pathogenic variants are enriched at hotspot positions.")
    elif ci_hi < 1:
        print("  Interpretation: CI excludes 1 → benign variants are enriched at hotspot positions (unexpected).")
    else:
        print("  Interpretation: CI includes 1 → no statistically significant enrichment at this sample size.")

    # 6. Chi-square test (reported for completeness if valid)
    chi2_result = None
    if use_chisq:
        chi2, p_chi2, dof, _ = stats.chi2_contingency(table, correction=False)
        print(f"\n--- Chi-square test (Pearson, df=1) ---")
        print(f"  χ²           : {chi2:.4f}")
        print(f"  p-value      : {p_chi2:.6e}")
        chi2_result = (chi2, p_chi2)

    # 7. Relative risk (as supplementary measure)
    total_hotspot     = a + c
    total_non_hotspot = b + d
    rr = (a / total_hotspot) / (b / total_non_hotspot) if total_non_hotspot > 0 else float("nan")
    print(f"\n--- Supplementary: Relative Risk (RR) ---")
    print(f"  P(pathogenic | hotspot)     : {a/total_hotspot:.4f}")
    print(f"  P(pathogenic | non-hotspot) : {b/total_non_hotspot:.4f}")
    print(f"  RR (hotspot vs non-hotspot) : {rr:.4f}")

    # 8. Save text report
    report_lines = [
        "TP53 Structural Hotspot Enrichment — Statistical Report",
        "=" * 56,
        "",
        "2×2 Contingency Table (observed counts):",
        f"  {'':20s} {'Hotspot':>10} {'Non-hotspot':>13} {'Total':>8}",
        f"  {'Pathogenic':20s} {a:>10} {b:>13} {a+b:>8}",
        f"  {'Benign':20s} {c:>10} {d:>13} {c+d:>8}",
        f"  {'Total':20s} {a+c:>10} {b+d:>13} {a+b+c+d:>8}",
        "",
        f"Expected counts (under H₀):",
        f"  Pathogenic | Hotspot     : {exp[0,0]:.2f}",
        f"  Pathogenic | Non-hotspot : {exp[0,1]:.2f}",
        f"  Benign     | Hotspot     : {exp[1,0]:.2f}",
        f"  Benign     | Non-hotspot : {exp[1,1]:.2f}",
        "",
        f"Fisher's exact test:",
        f"  OR (Fisher)   = {odds_ratio_fisher:.4f}",
        f"  p-value       = {p_fisher:.6e}   {'*' if p_fisher < ALPHA else 'ns'}",
        "",
        f"Woolf/logit OR with 95% CI:",
        f"  OR = {or_woolf:.4f}  (95% CI: {ci_lo:.4f} – {ci_hi:.4f})",
        "",
    ]
    if chi2_result:
        report_lines += [
            f"Chi-square test (Pearson, df=1):",
            f"  χ² = {chi2_result[0]:.4f}   p = {chi2_result[1]:.6e}",
            "",
        ]
    report_lines += [
        f"Relative risk (hotspot vs non-hotspot): {rr:.4f}",
        f"  P(pathogenic | hotspot)     = {a/total_hotspot:.4f}",
        f"  P(pathogenic | non-hotspot) = {b/total_non_hotspot:.4f}",
    ]

    report_path = os.path.join(OUTPUT_DIR, "statistical_results.txt")
    with open(report_path, "w") as fh:
        fh.write("\n".join(report_lines))
    print(f"\nText report saved → {report_path}")

    # Save contingency table as CSV
    ctab_df = pd.DataFrame(
        {"hotspot": [a, c], "non-hotspot": [b, d]},
        index=["pathogenic", "benign"],
    )
    ctab_path = os.path.join(OUTPUT_DIR, "contingency_table.csv")
    ctab_df.to_csv(ctab_path)
    print(f"Contingency table saved → {ctab_path}")

    # 9. Visualisation — heatmap of observed counts
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: annotated heatmap
    ax = axes[0]
    sns.heatmap(
        ctab_df.astype(float),
        annot=True, fmt=".0f", cmap="Blues",
        linewidths=0.5, linecolor="gray",
        ax=ax, cbar_kws={"label": "Count"},
    )
    ax.set_title(
        f"Observed counts\nFisher p = {p_fisher:.3e}  |  OR = {or_woolf:.2f} "
        f"(95% CI {ci_lo:.2f}–{ci_hi:.2f})",
        fontsize=9,
    )
    ax.set_xlabel("Hotspot membership", fontsize=11)
    ax.set_ylabel("ClinVar label", fontsize=11)

    # Right: mosaic-style proportional bar chart
    ax2 = axes[1]
    groups = ["Hotspot", "Non-hotspot"]
    path_frac = [a/(a+c) if (a+c) > 0 else 0,
                 b/(b+d) if (b+d) > 0 else 0]
    ben_frac  = [c/(a+c) if (a+c) > 0 else 0,
                 d/(b+d) if (b+d) > 0 else 0]

    # Width proportional to total count in each column
    total_hs  = a + c
    total_nhs = b + d
    grand     = total_hs + total_nhs
    widths    = [total_hs / grand, total_nhs / grand]
    # normalise to sum to 1.8 for spacing
    widths_plot = [w * 0.8 for w in widths]
    x_pos = [0, 0.9]

    for i, (grp, pf, bf, xp, wd) in enumerate(
        zip(groups, path_frac, ben_frac, x_pos, widths_plot)
    ):
        ax2.bar(xp, pf,  width=wd, color="#d62728", alpha=0.85)
        ax2.bar(xp, bf, bottom=pf, width=wd, color="#1f77b4", alpha=0.85)
        ax2.text(xp, pf/2, f"{pf:.1%}\n({a if i==0 else b})",
                 ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax2.text(xp, pf + bf/2, f"{bf:.1%}\n({c if i==0 else d})",
                 ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        [f"Hotspot\n(n={total_hs})", f"Non-hotspot\n(n={total_nhs})"],
        fontsize=10,
    )
    ax2.set_ylabel("Proportion", fontsize=11)
    ax2.set_title("Pathogenic/Benign proportions\nby hotspot membership", fontsize=10)
    ax2.set_ylim(0, 1)

    red_patch  = mpatches.Patch(color="#d62728", alpha=0.85, label="Pathogenic")
    blue_patch = mpatches.Patch(color="#1f77b4", alpha=0.85, label="Benign")
    ax2.legend(handles=[red_patch, blue_patch], loc="upper right", fontsize=9)

    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUT_DIR, "contingency_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Heatmap + mosaic plot saved → {heatmap_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
