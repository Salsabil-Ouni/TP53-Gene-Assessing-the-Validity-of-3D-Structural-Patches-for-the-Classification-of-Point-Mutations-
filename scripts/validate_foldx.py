"""
validate_foldx.py
=================
Script 2 of 5 — FoldX vs. FireprotDB validation

PURPOSE:
    Correlate FoldX predicted ∆∆G values (from the PositionScan output parsed
    by parse_positionscan.py) against experimental ∆∆G values from FireprotDB
    for the 34 single-point mutations available in fireprot_tp53_muta_ddg.csv.

    Computes:
        • Pearson correlation coefficient r and p-value
        • Spearman ρ (rank-based, more robust to outliers)
        • MAE and RMSE
        • Scatter plot with regression line

    NOTE on known limitations:
        Contact mutants like R273H (experimental ∆∆G ≈ 0.35) may be under-
        predicted by FoldX because they remove DNA contacts rather than
        destabilising the fold — this is a recognised, expected discrepancy
        and is annotated on the plot.

INPUT:
    fireprot_tp53_muta_ddg.csv          (experimental values, same directory)
    derived/foldx_per_mutation.csv      (FoldX ∆∆G, output of script 1)

OUTPUT:
    derived/foldx_validation.csv        — merged table used for correlation
    derived/foldx_validation_scatter.png

USAGE:
    python validate_foldx.py
"""

# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOL DEVIATION — FIREPROT ROLE
# ═══════════════════════════════════════════════════════════════════════════════
# The study protocol states that thermodynamic data from FireprotDB will be
# collected as the source of mutation stability information. In practice,
# FireprotDB contains only 35 entries for TP53, of which only 4 match exactly
# (by substitution code) to the FoldX PositionScan results. This is far too
# sparse to drive a domain-wide hotspot analysis across 194 positions.
#
# ACTUAL ROLE OF FIREPROT IN THIS PIPELINE:
#   FireprotDB (fireprot_tp53_muta_ddg.csv) is used ONLY in this script, as
#   experimental ground truth for post-hoc validation of FoldX accuracy. It
#   is NOT used for hotspot identification. The primary ΔΔG source for hotspot
#   prediction is FoldX PositionScan (see parse_positionscan.py).
#
# This is standard practice: compute broadly with a force field, validate
# against sparse experimental data where available. Reference: Gerasimavicius
# et al. (2020), Sci Rep, doi:10.1038/s41598-020-72404-w.
# ═══════════════════════════════════════════════════════════════════════════════

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FIREPROT_FILE  = "fireprot_tp53_muta_ddg.csv"
FOLDX_FILE     = os.path.join("derived", "foldx_per_mutation.csv")
OUTPUT_DIR     = "derived"

# Contact mutants: known to have low experimental ∆∆G because they lose
# DNA contacts, not fold stability. Annotated separately on the plot.
CONTACT_MUTANTS = {"R273H", "R248Q"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalise_mut_code(code: str) -> str:
    """
    Normalise a mutation code to  <WT_aa1><position><Mut_aa1>  e.g. R175H.
    Handles codes with or without chain suffix.
    """
    code = str(code).strip()
    # Strip chain suffix like _A
    if "_" in code:
        code = code.split("_")[0]
    return code.upper()


def ensure_derived():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_derived()

    print("=" * 60)
    print("FoldX vs FireprotDB Validation")
    print("=" * 60)

    # 1. Load experimental data
    if not os.path.exists(FIREPROT_FILE):
        print(f"[ERROR] Not found: {FIREPROT_FILE}")
        return
    exp_df = pd.read_csv(FIREPROT_FILE)
    exp_df.columns = exp_df.columns.str.strip()
    exp_df["mut_code"] = exp_df["SUBSTITUTION"].apply(normalise_mut_code)
    exp_df = exp_df.rename(columns={"DDG": "ddg_exp", "position": "position"})
    exp_df = exp_df[["mut_code", "ddg_exp", "position"]]
    print(f"\nFireprotDB entries loaded: {len(exp_df)}")
    print(exp_df.head())

    # 2. Load FoldX predictions
    if not os.path.exists(FOLDX_FILE):
        print(f"\n[ERROR] Not found: {FOLDX_FILE}")
        print("Run parse_positionscan.py first to generate derived/foldx_per_mutation.csv")
        return
    foldx_df = pd.read_csv(FOLDX_FILE)
    foldx_df["mut_code"] = foldx_df["mut_code"].apply(normalise_mut_code)
    print(f"\nFoldX predictions loaded: {len(foldx_df)} rows")

    # 3. Merge on normalised mutation code
    merged = pd.merge(
        exp_df,
        foldx_df[["mut_code", "ddg_foldx"]],
        on="mut_code",
        how="inner",
    )
    print(f"\nOverlapping mutations (matched): {len(merged)}")
    if len(merged) == 0:
        print("[WARN] No overlapping mutations found. Check that parse_positionscan.py ran correctly.")
        return

    # 4. If a mutation appears multiple times in FireprotDB (duplicate experiments),
    #    take the mean experimental ∆∆G.
    merged = (
        merged.groupby(["mut_code", "position"])
        .agg(ddg_exp=("ddg_exp", "mean"), ddg_foldx=("ddg_foldx", "first"))
        .reset_index()
    )
    print(f"After deduplication: {len(merged)} unique mutations")

    # 5. Print comparison table
    merged = merged.sort_values("mut_code")
    merged["is_contact"] = merged["mut_code"].isin(CONTACT_MUTANTS)
    print("\nMutation-level comparison:")
    print(merged[["mut_code", "position", "ddg_exp", "ddg_foldx", "is_contact"]].to_string(index=False))

    # 6. Correlations
    x = merged["ddg_exp"].values
    y = merged["ddg_foldx"].values

    pearson_r, pearson_p  = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    mae  = np.mean(np.abs(y - x))
    rmse = np.sqrt(np.mean((y - x) ** 2))

    print("\n--- Correlation statistics ---")
    print(f"  N mutations          : {len(merged)}")
    print(f"  Pearson  r           : {pearson_r:.4f}   (p = {pearson_p:.4e})")
    print(f"  Spearman ρ           : {spearman_r:.4f}   (p = {spearman_p:.4e})")
    print(f"  MAE  (kcal/mol)      : {mae:.4f}")
    print(f"  RMSE (kcal/mol)      : {rmse:.4f}")

    if pearson_r >= 0.7:
        print("  → Strong positive correlation: FoldX reliably captures stability trend.")
    elif pearson_r >= 0.4:
        print("  → Moderate correlation: FoldX broadly correct but some scatter.")
    else:
        print("  → Weak correlation — inspect contact mutant outliers and dataset size.")

    # 7. Save merged validation table
    merged_path = os.path.join(OUTPUT_DIR, "foldx_validation.csv")
    merged.to_csv(merged_path, index=False)
    print(f"\nValidation table saved → {merged_path}")

    # 8. Scatter plot
    fig, ax = plt.subplots(figsize=(7, 6))

    # Regular mutations
    mask_reg = ~merged["is_contact"]
    ax.scatter(
        merged.loc[mask_reg, "ddg_exp"],
        merged.loc[mask_reg, "ddg_foldx"],
        color="#1f77b4", s=60, zorder=3, label="Structural/other mutations",
    )
    # Contact mutants annotated separately
    mask_con = merged["is_contact"]
    ax.scatter(
        merged.loc[mask_con, "ddg_exp"],
        merged.loc[mask_con, "ddg_foldx"],
        color="#d62728", s=80, marker="^", zorder=4,
        label="Contact mutants (R248Q, R273H)",
    )

    # Annotate each point
    for _, row in merged.iterrows():
        ax.annotate(
            row["mut_code"],
            (row["ddg_exp"], row["ddg_foldx"]),
            textcoords="offset points", xytext=(5, 3),
            fontsize=7, color="#333333",
        )

    # Regression line
    m, b, *_ = stats.linregress(x, y)
    x_line = np.linspace(x.min() - 0.5, x.max() + 0.5, 100)
    ax.plot(x_line, m * x_line + b, color="gray", linewidth=1.2,
            linestyle="--", label=f"Linear fit (slope={m:.2f})")

    # Identity line
    lim = [min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 1]
    ax.plot(lim, lim, color="black", linewidth=0.8, linestyle=":", label="Identity (y = x)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    ax.set_xlabel("Experimental ∆∆G — FireprotDB (kcal/mol)", fontsize=11)
    ax.set_ylabel("Predicted ∆∆G — FoldX (kcal/mol)", fontsize=11)
    ax.set_title(
        f"FoldX vs FireprotDB  |  Pearson r = {pearson_r:.3f}  "
        f"Spearman ρ = {spearman_r:.3f}  (N={len(merged)})",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "foldx_validation_scatter.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Scatter plot saved → {plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
