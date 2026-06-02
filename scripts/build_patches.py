"""
build_patches.py
================
Script 3 of 5 — 3D structural patch construction

PURPOSE:
    For each hotspot residue identified by parse_positionscan.py, construct a
    3D structural patch centred on the Cα atom of that residue in chain A of
    tp53_Repair.pdb.

    A patch is defined as all residues that have at least one heavy atom within
    6 Å of the central Cα, found via BioPython NeighborSearch (KD-tree).

    For every residue in each patch the following physicochemical properties
    are encoded:
        • residue_name      — three-letter amino acid code
        • residue_aa1       — single-letter amino acid code
        • charge            — positive / negative / neutral
        • hydrophobicity    — Kyte-Doolittle scale value
        • relative_sasa     — relative solvent-accessible surface area
                              (Shrake-Rupley algorithm via BioPython, no
                               external binary required)
        • residue_size      — number of heavy atoms in the residue
        • distance_to_center— distance (Å) from neighbour Cα to central Cα

    IMPORTANT DESIGN NOTE:
        The patch encodes the WILD-TYPE microenvironment, not the post-mutation
        conformation. This is intentional (the hypothesis being tested is whether
        the WT structural context predicts pathogenicity) but is a limitation to
        acknowledge in the thesis.

    CHAIN NOTE:
        2OCJ has four identical chains A–D. Only chain A is used for patch
        construction (canonical reference chain). All extracted neighbours are
        also from chain A only.

INPUT:
    tp53_Repair.pdb                    (FoldX-repaired structure, same directory)
    derived/foldx_hotspots.csv         (output of parse_positionscan.py)

OUTPUT:
    derived/patch_profiles.csv         — one row per (hotspot_position, neighbour_position)
    derived/patch_summary.csv          — one row per hotspot: patch size, mean properties
    derived/patch_size_distribution.png

USAGE:
    python build_patches.py
    python build_patches.py --radius 6.0 --pdb tp53_Repair.pdb
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SCOPE: HOTSPOT POSITIONS ONLY (aligned with study protocol)
# ═══════════════════════════════════════════════════════════════════════════════
# Per the study protocol: "Les hotspots identifiés seront ensuite caractérisés
# par la construction de patches structuraux 3D." This script builds patches
# ONLY for hotspot positions identified by parse_positionscan.py, which is
# exactly what the protocol specifies.
#
# Outputs:  derived/patch_profiles.csv  (hotspot positions only)
#           derived/patch_summary.csv   (hotspot positions only)
# These are used for descriptive characterisation of hotspot microenvironments
# (Results Section 3.3 in the thesis).
#
# DESIGN NOTE — classify_patches.py scope differs:
#   The classification script (classify_patches.py) independently rebuilds
#   patches for ALL 194 scanned positions (both hotspot and non-hotspot).
#   This is necessary so the classifier can compare the two groups.
#   Those extended outputs go to derived/all_position_patches.csv and are
#   separate from the characterisation outputs produced here.
#
# PDB CHAIN: chain A of tp53_Repair.pdb (FoldX-repaired 2OCJ) is used
# exclusively. 2OCJ has four identical chains (A–D); chain A is the
# canonical reference. All neighbour searches are restricted to chain A.
# ═══════════════════════════════════════════════════════════════════════════════

import os
import warnings
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Bio import PDB
from Bio.PDB import NeighborSearch, Selection
from Bio.PDB.SASA import ShrakeRupley

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_PDB      = "tp53_Repair.pdb"
HOTSPOT_FILE     = os.path.join("derived", "foldx_hotspots.csv")
OUTPUT_DIR       = "derived"
CHAIN_ID         = "A"   # reference chain
PATCH_RADIUS     = 6.0   # Å from central Cα to any heavy atom of neighbour


# ---------------------------------------------------------------------------
# Physicochemical property tables
# ---------------------------------------------------------------------------
# Kyte-Doolittle hydrophobicity scale (J. Mol. Biol. 157:105, 1982)
KD_HYDROPHOBICITY = {
    "ALA": 1.8,  "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
    "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
    "LEU": 3.8,  "LYS": -3.9, "MET": 1.9,  "PHE": 2.8,  "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}

# Formal charge at physiological pH (simplified)
CHARGE = {
    "ARG": "positive", "HIS": "positive", "LYS": "positive",
    "ASP": "negative", "GLU": "negative",
}

ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Approximate heavy atom counts per residue (backbone N,CA,C,O always = 4)
HEAVY_ATOM_COUNTS = {
    "ALA": 5,  "ARG": 11, "ASN": 8,  "ASP": 8,  "CYS": 6,
    "GLN": 9,  "GLU": 9,  "GLY": 4,  "HIS": 10, "ILE": 8,
    "LEU": 8,  "LYS": 9,  "MET": 8,  "PHE": 11, "PRO": 7,
    "SER": 6,  "THR": 7,  "TRP": 14, "TYR": 12, "VAL": 7,
}

# Maximum relative SASA (Å²) values — Tien et al., 2013
MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 223.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}


# ---------------------------------------------------------------------------
# SASA computation — BioPython ShrakeRupley (no external binary needed)
# ---------------------------------------------------------------------------
def compute_sasa(structure, chain_id: str) -> dict:
    """
    Compute per-residue absolute SASA (Å²) using BioPython's built-in
    Shrake-Rupley algorithm.  No external binary required.

    Returns dict {resnum: relative_sasa (0-1)} for chain_id.
    Relative SASA = absolute SASA / max_ASA for that residue type
    (Tien et al., 2013 theoretical maxima).
    """
    sr = ShrakeRupley()
    sr.compute(structure, level="R")   # residue-level SASA

    sasa_map = {}
    model = structure[0]
    chain = model[chain_id]
    for res in chain.get_residues():
        if res.get_id()[0] != " ":
            continue   # skip HETATM / water
        resname = res.get_resname().strip()
        resnum  = res.get_id()[1]
        abs_sasa = res.sasa
        max_asa  = MAX_ASA.get(resname, 200.0)   # fallback for unknowns
        rel_sasa = min(abs_sasa / max_asa, 1.0)  # cap at 1.0
        sasa_map[resnum] = round(rel_sasa, 3)

    print(f"  ShrakeRupley SASA computed for {len(sasa_map)} residues in chain {chain_id}.")
    return sasa_map


def get_relative_sasa(resname: str, resnum: int, sasa_map: dict) -> float:
    """Return relative SASA (0-1) from the precomputed ShrakeRupley map."""
    return sasa_map.get(resnum, 0.40)   # 0.40 fallback if residue not found


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------
def build_patch(structure, chain_id: str, center_resnum: int,
                radius: float, dssp_map: dict) -> list:
    """
    Centre on Cα of chain_id:center_resnum; collect all residues in chain_id
    that have any heavy atom within `radius` Å of that Cα.

    Returns a list of dicts, one per neighbour residue.
    """
    model = structure[0]
    chain = model[chain_id]

    # Find central Cα
    try:
        center_res = chain[center_resnum]
    except KeyError:
        print(f"  [WARN] Residue {center_resnum} not found in chain {chain_id}.")
        return []

    if "CA" not in center_res:
        print(f"  [WARN] No Cα atom in residue {center_resnum}.")
        return []

    ca_center = center_res["CA"].get_vector()

    # Collect all heavy atoms from chain A
    all_atoms = [
        atom for res in chain.get_residues()
        if res.get_id()[0] == " "  # exclude HETATM / water
        for atom in res.get_atoms()
        if atom.element != "H" and atom.element is not None
    ]

    ns = NeighborSearch(all_atoms)
    # Search around the central Cα coordinate
    ca_coord = np.array([ca_center[0], ca_center[1], ca_center[2]])
    nearby_atoms = ns.search(ca_coord, radius, level="A")

    # Collect unique residues from nearby atoms
    seen = set()
    patch_rows = []
    for atom in nearby_atoms:
        res  = atom.get_parent()
        rnum = res.get_id()[1]
        if rnum in seen:
            continue
        if res.get_id()[0] != " ":
            continue   # skip HETATM
        seen.add(rnum)

        resname = res.get_resname().strip()
        aa1     = ONE_LETTER.get(resname, "X")

        # Distance from this residue's Cα to the central Cα
        if "CA" in res:
            nb_ca   = res["CA"].get_vector()
            dist    = round(float((nb_ca - ca_center).norm()), 3)
        else:
            dist = None

        # Heavy atom count (actual count from structure)
        heavy_count = sum(
            1 for a in res.get_atoms()
            if a.element != "H" and a.element is not None
        )

        patch_rows.append(
            {
                "center_position":  center_resnum,
                "neighbour_position": rnum,
                "is_center":        int(rnum == center_resnum),
                "residue_name":     resname,
                "residue_aa1":      aa1,
                "charge":           CHARGE.get(resname, "neutral"),
                "hydrophobicity":   KD_HYDROPHOBICITY.get(resname, 0.0),
                "relative_sasa":    get_relative_sasa(resname, rnum, dssp_map),
                "residue_size":     heavy_count,
                "distance_to_center": dist,
            }
        )

    return patch_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(pdb_file: str, radius: float):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("3D Structural Patch Construction")
    print("=" * 60)

    # Load hotspot list
    if not os.path.exists(HOTSPOT_FILE):
        print(f"[ERROR] Not found: {HOTSPOT_FILE}")
        print("Run parse_positionscan.py first.")
        return
    hotspots = pd.read_csv(HOTSPOT_FILE)
    hotspot_positions = sorted(hotspots["position"].astype(int).tolist())
    print(f"\nHotspot positions loaded: {len(hotspot_positions)}")
    print(f"  {hotspot_positions}")

    # Load PDB structure
    if not os.path.exists(pdb_file):
        print(f"[ERROR] PDB file not found: {pdb_file}")
        return
    parser_pdb = PDB.PDBParser(QUIET=True)
    structure  = parser_pdb.get_structure("tp53", pdb_file)
    print(f"\nStructure loaded: {pdb_file}")
    print(f"  Using chain {CHAIN_ID} (2OCJ canonical reference chain)")
    print(f"  Patch radius: {radius} Å (Cα-centred, any heavy atom of neighbour)")

    # Compute SASA via BioPython ShrakeRupley (pure Python, no binary needed)
    dssp_map = compute_sasa(structure, CHAIN_ID)

    # Build patches for every hotspot
    all_patch_rows = []
    summary_rows   = []

    for pos in hotspot_positions:
        print(f"\n  Building patch for residue {pos} …", end=" ")
        rows = build_patch(structure, CHAIN_ID, pos, radius, dssp_map)
        if not rows:
            print("SKIPPED (no Cα or residue not found).")
            continue
        all_patch_rows.extend(rows)

        # Summary statistics for this patch
        patch_df = pd.DataFrame(rows)
        n_nb     = len(patch_df)
        mean_hyd = patch_df["hydrophobicity"].mean()
        frac_pos = (patch_df["charge"] == "positive").mean()
        frac_neg = (patch_df["charge"] == "negative").mean()
        frac_neu = (patch_df["charge"] == "neutral").mean()
        mean_sasa= patch_df["relative_sasa"].mean()
        mean_size= patch_df["residue_size"].mean()

        print(f"{n_nb} neighbours")
        summary_rows.append(
            {
                "position":          pos,
                "patch_size":        n_nb,
                "mean_hydrophobicity": round(mean_hyd, 3),
                "frac_positive":     round(frac_pos, 3),
                "frac_negative":     round(frac_neg, 3),
                "frac_neutral":      round(frac_neu, 3),
                "mean_relative_sasa":round(mean_sasa, 3),
                "mean_residue_size": round(mean_size, 1),
            }
        )

    # Save outputs
    profile_df = pd.DataFrame(all_patch_rows)
    profile_path = os.path.join(OUTPUT_DIR, "patch_profiles.csv")
    profile_df.to_csv(profile_path, index=False)
    print(f"\nPatch profiles saved → {profile_path}  ({len(profile_df)} rows)")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "patch_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Patch summary saved → {summary_path}")

    print("\n--- Patch summary ---")
    print(summary_df.to_string(index=False))

    # Visualisation — patch size distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary_df["position"].astype(str), summary_df["patch_size"],
           color="#1f77b4", alpha=0.85)
    ax.set_xlabel("Hotspot residue position", fontsize=11)
    ax.set_ylabel(f"Patch size (residues within {radius} Å)", fontsize=11)
    ax.set_title("3D structural patch sizes — TP53 hotspot residues", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "patch_size_distribution.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Patch size plot saved → {plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Build 3D structural patches for hotspot residues.")
    arg_parser.add_argument("--pdb",    default=DEFAULT_PDB,  help="FoldX-repaired PDB file")
    arg_parser.add_argument("--radius", type=float, default=PATCH_RADIUS,
                            help=f"Patch radius in Å (default: {PATCH_RADIUS})")
    args = arg_parser.parse_args()
    main(args.pdb, args.radius)
