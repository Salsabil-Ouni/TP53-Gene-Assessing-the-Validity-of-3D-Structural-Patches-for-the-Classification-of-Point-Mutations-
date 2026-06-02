"""
parse_positionscan.py
=====================
Script 1 of 5 — FoldX PositionScan output parser

PURPOSE:
    Parse the FoldX PositionScan output file(s), compute per-position mean ∆∆G
    across all 19 substitutions, and flag residues whose mean ∆∆G ≥ 1.5 kcal/mol
    as structural hotspots.

INPUT:
    PS_tp53_Repair_scanning_output.txt   (FoldX PositionScan output, same directory)

OUTPUT:
    derived/foldx_per_mutation.csv   — one row per (position, substitution)
    derived/foldx_hotspots.csv       — one row per hotspot position (mean ∆∆G ≥ 1.5)

FoldX PositionScan output format (FoldX 5):
    Each line:  MutCode  total_energy  <energy_terms...>
    MutCode format: <WT_aa1><chain><resnum><Mut_aa1>  e.g.  FA96A  or  FA96A_A
    A WT "mutation" (same aa → same aa) gives the wild-type energy for that position.
    ∆∆G = E_mutant − E_wildtype_at_same_position

USAGE:
    python parse_positionscan.py
    python parse_positionscan.py --input my_scan_output.txt --threshold 1.5
"""

# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOL ALIGNMENT — FREE-ENERGY HOTSPOT PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
# This script directly implements section 4 of the study protocol:
#   "Une prédiction des hotspots protéiques sera réalisée à l'aide d'un
#    algorithme d'estimation de l'énergie libre."
#
# HOTSPOT DEFINITION: a position is a structural stability hotspot if its
# MEAN ΔΔG across all 19 non-synonymous substitutions at that position is
# ≥ 1.5 kcal/mol. The 1.5 kcal/mol threshold is standard (Gerasimavicius
# et al., 2020) and is ~25% of the total folding stability of the TP53
# DNA-binding domain (~6 kcal/mol, Bullock et al., 1997).
#
# Using the MEAN (not individual substitution ΔΔG) reflects structural
# fragility of the position regardless of which amino acid replaces it.
# A position with mean ΔΔG ≥ 1.5 is destabilising on average — it is a
# structurally constrained site where most substitutions are damaging.
#
# FIREPROT NOTE: The protocol also mentions FireprotDB as the thermodynamic
# data source. FireprotDB is used ONLY for post-hoc validation in
# validate_foldx.py, not here. See that script for the full rationale.
# ═══════════════════════════════════════════════════════════════════════════════

import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_INPUT  = "PS_tp53_Repair_scanning_output.txt"
DEFAULT_THRESH = 1.5   # kcal/mol — destabilising threshold
OUTPUT_DIR     = "derived"

ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # FoldX protonated histidine variants
    "H2S": "H", "H1S": "H", "HID": "H", "HIE": "H", "HIP": "H",
    "HSD": "H", "HSE": "H", "HSP": "H",
}

# FoldX PositionScan uses lowercase letters for non-standard amino acids:
#   'e' = histidine (protonated form, HIE/HIP) — maps to H
#   'o' = hydroxyproline — not a standard AA, skip
FOLDX_SPECIAL = {"e": "H"}  # 'o' is hydroxyproline, intentionally omitted


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_derived():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_mutation_code(code: str):
    """
    Accept any of these FoldX mutation code formats and return
    (wt_aa1, chain, resnum, mut_aa1):

        SERA96S        — 3-letter WT (uppercase) + chain + position + 1-letter mut
                         (this is the tp532 PositionScan format)
        FA96A          — 1-letter WT + chain + position + 1-letter mut
        FA96A_A        — chain after underscore
        Phe96Ala       — mixed-case three-letter codes
        Phe96Ala_A
    """
    code = code.strip().rstrip(";").strip()

    # Strip optional chain suffix  (_A, _B …)
    chain = None
    chain_match = re.search(r"_([A-Z])$", code)
    if chain_match:
        chain = chain_match.group(1)
        code = code[: chain_match.start()]

    def resolve_mut(raw):
        """Map FoldX special lowercase codes; uppercase letters pass through unchanged."""
        if raw.islower():
            # Lowercase = FoldX special code (e.g. 'e' = His, 'o' = hydroxyproline)
            return FOLDX_SPECIAL.get(raw)  # None means skip this entry
        return raw.upper()

    # ---- Format 1: SERA96S or H2SA115G or SERA96e (lowercase = special AA)
    # (2-3 uppercase letters/digits + 1-letter chain + digits + 1-letter/special mut)
    tp532_fmt = re.match(r"^([A-Z][A-Z0-9]{2})([A-Z])(\d+)([A-Za-z])$", code)
    if tp532_fmt:
        wt3, ch, resnum, mut_raw = tp532_fmt.groups()
        wt_aa  = ONE_LETTER.get(wt3.upper())
        mut_aa = resolve_mut(mut_raw)
        chain  = chain or ch
        return wt_aa, chain, int(resnum), mut_aa

    # ---- Format 2: Phe96Ala  (mixed-case three-letter codes)
    three_letter = re.match(
        r"([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$", code, re.IGNORECASE
    )
    if three_letter:
        wt3, resnum, mut3 = three_letter.groups()
        wt_aa  = ONE_LETTER.get(wt3.upper())
        mut_aa = ONE_LETTER.get(mut3.upper())
        return wt_aa, chain, int(resnum), mut_aa

    # ---- Format 3: FA96A  (1-letter WT + optional-chain + position + 1-letter mut)
    single = re.match(r"([A-Z])([A-Z]?)(\d+)([A-Za-z])$", code)
    if single:
        g = single.groups()
        if g[1] == "":
            wt_aa, _, resnum, mut_raw = g
        else:
            wt_aa  = g[0]
            chain  = chain or g[1]
            resnum = g[2]
            mut_raw = g[3]
        return wt_aa, chain, int(resnum), resolve_mut(mut_raw)

    return None, chain, None, None


def read_positionscan_file(filepath: str) -> pd.DataFrame:
    """
    Read a FoldX PositionScan output file.
    Returns a DataFrame with columns:
        mut_code, wt_aa, chain, position, mut_aa, total_energy
    """
    rows = []
    print(f"  Reading: {filepath}")

    with open(filepath, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Split on whitespace; some files use tab, some spaces
            parts = line.split()
            if len(parts) < 2:
                continue
            mut_code = parts[0].rstrip(";")
            try:
                total_energy = float(parts[1])
            except ValueError:
                # Likely a header row
                continue

            wt_aa, chain, position, mut_aa = parse_mutation_code(mut_code)
            if position is None:
                print(f"    [WARN] Could not parse mutation code: {mut_code}")
                continue
            if mut_aa is None:
                continue  # FoldX special code with no standard-AA mapping (e.g. 'o')

            rows.append(
                {
                    "mut_code":     mut_code,
                    "wt_aa":        wt_aa,
                    "chain":        chain,
                    "position":     position,
                    "mut_aa":       mut_aa,
                    "total_energy": total_energy,
                }
            )

    return pd.DataFrame(rows)


def compute_ddg(df: pd.DataFrame) -> pd.DataFrame:
    """
    FoldX PositionScan already outputs ∆∆G relative to WT
    (the WT→WT row has ∆∆G = 0).  So total_energy IS ∆∆G here.

    We only need to:
    - Filter to chain A (canonical reference; 2OCJ has 4 chains — A/B scanned)
    - Drop WT self-substitution rows (wt_aa == mut_aa)
    - Re-label the column as ddg_foldx
    """
    # Use chain A only
    if df["chain"].notna().any():
        df_a = df[df["chain"] == "A"].copy()
        n_dropped = len(df) - len(df_a)
        if n_dropped > 0:
            print(f"    Dropped {n_dropped} rows from chains other than A.")
    else:
        df_a = df.copy()

    # Drop WT self-substitutions (wt_aa == mut_aa, ddg ~ 0)
    df_a = df_a[df_a["wt_aa"] != df_a["mut_aa"]].copy()

    df_a = df_a.rename(columns={"total_energy": "ddg_foldx"})
    df_a["mut_code"]  = df_a["wt_aa"].fillna("X") + df_a["position"].astype(str) + df_a["mut_aa"].fillna("X")
    df_a["ddg_foldx"] = df_a["ddg_foldx"].round(4)

    return df_a[["position", "chain", "wt_aa", "mut_aa", "mut_code", "ddg_foldx"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(input_file: str, threshold: float):
    ensure_derived()

    print("=" * 60)
    print("FoldX PositionScan Parser")
    print("=" * 60)

    if not os.path.exists(input_file):
        print(f"\n[ERROR] Input file not found: {input_file}")
        print(
            "Expected FoldX PositionScan output file.\n"
            "Run FoldX with:  --command=PositionScan  --pdb=tp53_Repair.pdb\n"
            "                 --positions=FA96A  (or use a positions file)\n"
        )
        return

    # 1. Read raw FoldX output
    raw_df = read_positionscan_file(input_file)
    print(f"\nTotal lines parsed: {len(raw_df)}")
    print(f"Unique positions  : {raw_df['position'].nunique()}")
    print(f"Unique chains     : {raw_df['chain'].unique()}")

    # 2. Compute ∆∆G relative to WT at each position
    print("\nComputing ∆∆G = E_mutant − E_WT at each position …")
    ddg_df = compute_ddg(raw_df)
    print(f"Mutation-level rows after ∆∆G calculation: {len(ddg_df)}")

    # 3. Save per-mutation table
    per_mut_path = os.path.join(OUTPUT_DIR, "foldx_per_mutation.csv")
    ddg_df.to_csv(per_mut_path, index=False)
    print(f"Saved per-mutation table → {per_mut_path}")

    # 4. Compute mean ∆∆G per position
    pos_summary = (
        ddg_df.groupby("position")
        .agg(
            wt_aa=("wt_aa", "first"),
            mean_ddg=("ddg_foldx", "mean"),
            max_ddg=("ddg_foldx", "max"),
            n_substitutions=("ddg_foldx", "count"),
        )
        .reset_index()
    )
    pos_summary["mean_ddg"] = pos_summary["mean_ddg"].round(4)
    pos_summary["max_ddg"]  = pos_summary["max_ddg"].round(4)
    pos_summary = pos_summary.sort_values("position")

    print(f"\nPer-position summary ({len(pos_summary)} positions):")
    print(pos_summary.describe())

    # 5. Apply hotspot threshold
    hotspots = pos_summary[pos_summary["mean_ddg"] >= threshold].copy()
    non_hotspots = pos_summary[pos_summary["mean_ddg"] < threshold].copy()
    hotspots["is_hotspot"] = 1
    non_hotspots["is_hotspot"] = 0

    all_positions = pd.concat([hotspots, non_hotspots]).sort_values("position")
    all_positions_path = os.path.join(OUTPUT_DIR, "foldx_position_summary.csv")
    all_positions.to_csv(all_positions_path, index=False)
    print(f"\nAll positions with hotspot flag → {all_positions_path}")

    hotspot_path = os.path.join(OUTPUT_DIR, "foldx_hotspots.csv")
    hotspots.to_csv(hotspot_path, index=False)
    print(f"Hotspot positions ({len(hotspots)}) → {hotspot_path}")

    print(f"\n--- Hotspot summary (mean ∆∆G ≥ {threshold} kcal/mol) ---")
    print(f"  Total positions scanned : {len(pos_summary)}")
    print(f"  Hotspot positions       : {len(hotspots)}")
    print(f"  Non-hotspot positions   : {len(non_hotspots)}")
    print(f"\n  Hotspot residues:")
    for _, row in hotspots.sort_values("mean_ddg", ascending=False).iterrows():
        print(
            f"    {row['wt_aa']}{int(row['position']):<5}  "
            f"mean ∆∆G = {row['mean_ddg']:6.2f}  max = {row['max_ddg']:6.2f}"
        )

    # Note known contact mutants that may NOT appear as hotspots
    contact_mutants = [248, 273]
    for pos in contact_mutants:
        if pos in pos_summary["position"].values:
            val = pos_summary.loc[pos_summary["position"] == pos, "mean_ddg"].values[0]
            flag = "HOTSPOT" if val >= threshold else "not hotspot (expected — contact mutant)"
            print(f"\n  [NOTE] Position {pos} (contact mutant): mean ∆∆G = {val:.2f} → {flag}")

    # 6. Visualisation — histogram of mean ∆∆G per position
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        pos_summary["position"],
        pos_summary["mean_ddg"],
        width=1.0,
        color=["#d62728" if v >= threshold else "#1f77b4" for v in pos_summary["mean_ddg"]],
        edgecolor="none",
        alpha=0.85,
    )
    ax.axhline(threshold, color="black", linewidth=1.2, linestyle="--",
               label=f"Hotspot threshold ({threshold} kcal/mol)")
    ax.set_xlabel("Residue position (TP53 core domain)", fontsize=11)
    ax.set_ylabel("Mean ∆∆G (kcal/mol)", fontsize=11)
    ax.set_title("FoldX mean ∆∆G per position — TP53 core domain (residues 96–289)", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "foldx_mean_ddg_per_position.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved → {plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse FoldX PositionScan output.")
    parser.add_argument(
        "--input", default=DEFAULT_INPUT,
        help=f"Path to FoldX PositionScan output file (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESH,
        help=f"Hotspot threshold in kcal/mol (default: {DEFAULT_THRESH})"
    )
    args = parser.parse_args()
    main(args.input, args.threshold)
