"""
map_variants.py
===============
Script 4 of 5 — ClinVar variant to hotspot mapping

PURPOSE:
    Merge the ClinVar variant table (tp53_clinvar_labeled.csv) with the FoldX
    hotspot position list (derived/foldx_hotspots.csv) to assign each variant
    a binary hotspot membership flag (1 = hotspot position, 0 = non-hotspot).

    Also merges in position-level FoldX mean ∆∆G for downstream analysis and
    produces a quick summary of how pathogenic/benign variants distribute across
    hotspot vs non-hotspot positions.

INPUT:
    tp53_clinvar_labeled.csv           (1,374 ClinVar variants, same directory)
    derived/foldx_hotspots.csv         (output of parse_positionscan.py)
    derived/foldx_position_summary.csv (all positions with hotspot flag,
                                        output of parse_positionscan.py)

OUTPUT:
    derived/variants_with_hotspot.csv  — ClinVar table + hotspot flag + mean ∆∆G
    derived/contingency_counts.csv     — 2×2 counts for statistical_analysis.py

USAGE:
    python map_variants.py
"""

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET NOTE — 1,374 VARIANTS IN, 866 IN DOMAIN (96–289)
# ═══════════════════════════════════════════════════════════════════════════════
# This script merges all 1,374 ClinVar-labeled variants with the FoldX hotspot
# list. Variants outside residues 96–289 receive is_hotspot=0 because the
# PositionScan never covered those positions — this is correct for this script
# (mapping), but the downstream statistical_analysis.py MUST filter to the
# domain before building its contingency table (which it does: positions 96–289
# only). See statistical_analysis.py for the detailed rationale.
# ═══════════════════════════════════════════════════════════════════════════════

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CLINVAR_FILE    = "tp53_clinvar_labeled.csv"
HOTSPOT_FILE    = os.path.join("derived", "foldx_hotspots.csv")
POS_SUMMARY     = os.path.join("derived", "foldx_position_summary.csv")
OUTPUT_DIR      = "derived"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("ClinVar Variant → Hotspot Mapping")
    print("=" * 60)

    # 1. Load ClinVar data
    if not os.path.exists(CLINVAR_FILE):
        print(f"[ERROR] Not found: {CLINVAR_FILE}")
        return
    clinvar = pd.read_csv(CLINVAR_FILE)
    clinvar.columns = clinvar.columns.str.strip()
    print(f"\nClinVar variants loaded: {len(clinvar)}")
    print(f"  Columns: {clinvar.columns.tolist()}")

    # Ensure position column is integer
    clinvar["position"] = clinvar["position"].astype(int)

    # Count label breakdown
    label_counts = clinvar["label"].value_counts()
    print(f"\n  Label distribution:")
    for lbl, cnt in label_counts.items():
        print(f"    {lbl:<12}: {cnt}")

    # 2. Load hotspot list
    if not os.path.exists(HOTSPOT_FILE):
        print(f"[ERROR] Not found: {HOTSPOT_FILE}")
        print("Run parse_positionscan.py first.")
        return
    hotspots = pd.read_csv(HOTSPOT_FILE)
    hotspot_positions = set(hotspots["position"].astype(int).tolist())
    print(f"\nHotspot positions: {len(hotspot_positions)}")
    print(f"  {sorted(hotspot_positions)}")

    # 3. Load full position summary for mean ∆∆G annotation
    mean_ddg_map = {}
    if os.path.exists(POS_SUMMARY):
        pos_df = pd.read_csv(POS_SUMMARY)
        pos_df["position"] = pos_df["position"].astype(int)
        mean_ddg_map = dict(zip(pos_df["position"], pos_df["mean_ddg"]))
        print(f"\nPosition-level ∆∆G values loaded for {len(mean_ddg_map)} positions.")
    else:
        print(f"\n[WARN] {POS_SUMMARY} not found; mean ∆∆G will not be annotated.")

    # 4. Assign hotspot membership to each variant
    clinvar["is_hotspot"]  = clinvar["position"].apply(
        lambda p: 1 if p in hotspot_positions else 0
    )
    clinvar["mean_ddg_foldx"] = clinvar["position"].map(mean_ddg_map)

    print(f"\nHotspot membership assigned:")
    print(f"  Variants at hotspot positions    : {clinvar['is_hotspot'].sum()}")
    print(f"  Variants at non-hotspot positions: {(clinvar['is_hotspot'] == 0).sum()}")

    # 5. Cross-tabulation: label × hotspot membership
    print("\n--- 2×2 cross-tabulation: label × is_hotspot ---")
    ctab = pd.crosstab(
        clinvar["label"],
        clinvar["is_hotspot"],
        margins=True,
        margins_name="Total",
    )
    ctab.columns = [f"is_hotspot={c}" if c != "Total" else "Total" for c in ctab.columns]
    print(ctab.to_string())

    # Fractions within each hotspot group
    inner_ctab = pd.crosstab(clinvar["label"], clinvar["is_hotspot"], normalize="columns")
    inner_ctab.columns = ["non-hotspot", "hotspot"]
    print("\nRow fractions within each hotspot group:")
    print(inner_ctab.round(3).to_string())

    # 6. Save outputs
    merged_path = os.path.join(OUTPUT_DIR, "variants_with_hotspot.csv")
    clinvar.to_csv(merged_path, index=False)
    print(f"\nMerged variant table saved → {merged_path}")

    # Save 2×2 contingency counts for statistical_analysis.py
    # Format: pathogenic_hotspot, pathogenic_non, benign_hotspot, benign_non
    path_hot   = int(((clinvar["label"] == "pathogenic") & (clinvar["is_hotspot"] == 1)).sum())
    path_non   = int(((clinvar["label"] == "pathogenic") & (clinvar["is_hotspot"] == 0)).sum())
    benign_hot = int(((clinvar["label"] == "benign")     & (clinvar["is_hotspot"] == 1)).sum())
    benign_non = int(((clinvar["label"] == "benign")     & (clinvar["is_hotspot"] == 0)).sum())

    contingency_df = pd.DataFrame(
        {
            "group":               ["pathogenic", "pathogenic", "benign", "benign"],
            "hotspot_membership":  ["hotspot",    "non-hotspot","hotspot","non-hotspot"],
            "count":               [path_hot, path_non, benign_hot, benign_non],
        }
    )
    cont_path = os.path.join(OUTPUT_DIR, "contingency_counts.csv")
    contingency_df.to_csv(cont_path, index=False)
    print(f"Contingency counts saved → {cont_path}")

    print(f"\n  Pathogenic at hotspot     : {path_hot}")
    print(f"  Pathogenic at non-hotspot : {path_non}")
    print(f"  Benign at hotspot         : {benign_hot}")
    print(f"  Benign at non-hotspot     : {benign_non}")

    # 7. Visualisation — stacked bar: hotspot vs non-hotspot, coloured by label
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: absolute counts
    data = {
        "Hotspot":     [path_hot,  benign_hot],
        "Non-hotspot": [path_non,  benign_non],
    }
    categories = ["Pathogenic", "Benign"]
    x = np.arange(len(categories))
    w = 0.35
    ax = axes[0]
    ax.bar(x - w/2, [path_hot,  benign_hot], w, label="Hotspot",     color="#d62728", alpha=0.85)
    ax.bar(x + w/2, [path_non, benign_non],  w, label="Non-hotspot", color="#1f77b4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Number of variants", fontsize=11)
    ax.set_title("Variant counts by label and hotspot membership", fontsize=10)
    ax.legend()

    # Right panel: fraction pathogenic at hotspot vs non-hotspot
    frac_hotspot = path_hot / (path_hot + benign_hot) if (path_hot + benign_hot) > 0 else 0
    frac_non     = path_non / (path_non + benign_non) if (path_non + benign_non) > 0 else 0
    ax2 = axes[1]
    bars = ax2.bar(
        ["Hotspot", "Non-hotspot"],
        [frac_hotspot, frac_non],
        color=["#d62728", "#1f77b4"], alpha=0.85, width=0.4,
    )
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Fraction pathogenic", fontsize=11)
    ax2.set_title("Fraction of pathogenic variants\nby hotspot membership", fontsize=10)
    for bar, frac in zip(bars, [frac_hotspot, frac_non]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{frac:.2%}", ha="center", fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "variant_hotspot_distribution.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nDistribution plot saved → {plot_path}")

    # 8. Pathogenicity enrichment at well-known positions (sanity check)
    print("\n--- Sanity check: known pathogenic hotspots ---")
    known = [175, 245, 248, 249, 273, 282]
    for pos in known:
        sub = clinvar[clinvar["position"] == pos]
        if sub.empty:
            print(f"  Position {pos}: no ClinVar variants")
            continue
        n_path  = (sub["label"] == "pathogenic").sum()
        n_ben   = (sub["label"] == "benign").sum()
        is_hot  = "hotspot" if pos in hotspot_positions else "non-hotspot"
        mean_ddg_val = mean_ddg_map.get(pos, float("nan"))
        print(
            f"  {pos:<5} ({is_hot:<12})  "
            f"path={n_path}  benign={n_ben}  "
            f"mean ∆∆G={mean_ddg_val:.2f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
