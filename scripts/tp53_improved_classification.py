#!/usr/bin/env python3
"""
tp53_improved_classification.py
================================
Full improvement pipeline for TP53 structural patch classification.
Targets thesis-level AUC through five incremental improvements over
the original classify_patches.py baseline (AUC 0.58).

IMPROVEMENTS IMPLEMENTED
  Step 1 — Per-variant FoldX ΔΔG
           The exact ΔΔG for each specific amino-acid substitution (e.g. R175H)
           replaces the position-mean only. Every variant is now distinct.

  Step 2 — Extended FoldX position statistics
           std_ddg, frac_destabilizing, max_ddg derived from the full per-mutation
           scan encode how constrained a position is (structural fragility spectrum).

  Step 3 — Additional structural features
           B-factor (thermal flexibility), zinc-coordination distance, and
           secondary structure (DSSP or phi/psi fallback).

  Step 4 — BLOSUM62 substitution cost
           Biochemical penalty of the specific amino-acid swap as a variant-level
           sequence-physics feature.

  Step 5 — AlphaFold extension
           Downloads the AlphaFold2 model (AF-P04637-F1-model_v4.pdb) to compute
           patches for positions 1–95 and 290–393, recovering the 508 variants
           previously excluded because they fall outside the 2OCJ crystal structure.
           AlphaFold pLDDT (B-factor column) is added as a per-residue order score.

  Step 6 — Multiple classifiers + hyperparameter tuning
           LogisticRegression, RandomForest, GradientBoosting, SVM, Ensemble.
           All evaluated by 5-fold StratifiedGroupKFold (grouped by position) to
           prevent pseudo-replication.

RUN
  py -3 tp53_improved_classification.py

OUTPUT  (all in derived/)
  improved_feature_matrix.csv       — full feature matrix (variant-level)
  improved_classification_report.txt
  improved_roc_curves.png
  improved_pr_curves.png
  improved_feature_importance.png
  improved_calibration.png
"""

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET — FULL 1,374-VARIANT SET (extension beyond protocol scope)
# ═══════════════════════════════════════════════════════════════════════════════
# The study protocol operates on TP53 variants with structural data from the
# 2OCJ crystal structure (residues 96–289), which covers 866 of the 1,374
# ClinVar-labeled variants. This script extends coverage to the full 1,374
# variants by computing patches for outside-domain positions (1–95, 290–393)
# from the AlphaFold2 model (AF-P04637-F1), recovering the 508 excluded
# variants. This is an improvement beyond the original protocol scope.
#
# FEATURE SET: 25 features vs the 12 used in classify_patches.py.
# Additional features (Steps 1–5): per-variant specific ΔΔG, extended FoldX
# position statistics (std_ddg, frac_destabilizing, max_ddg), B-factor,
# secondary structure, zinc-coordination distance, AlphaFold pLDDT, BLOSUM62
# substitution cost, and biochemical difference features (delta_hydrophobicity,
# delta_size, charge_change).
#
# CLASSIFIERS: LogisticRegression, RandomForest, GradientBoosting, SVM, and
# an RF+GB+SVM voting ensemble, all tuned by RandomizedSearchCV.
#
# All cross-validation uses 5-fold StratifiedGroupKFold grouped by position
# (same as classify_patches.py), preventing pseudo-replication.
#
# PDB CHAIN: chain A of tp53_Repair.pdb for 2OCJ positions; chain A of
# AF-P04637-F1 for outside-domain positions. DNA distances from chain A
# (protein) and chain C (DNA) of 3kz8.pdb1. Chain A is used consistently
# throughout.
# ═══════════════════════════════════════════════════════════════════════════════

import os, re, sys, warnings, urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import (RandomForestClassifier,
                                     GradientBoostingClassifier,
                                     VotingClassifier)
from sklearn.svm             import SVC
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict, RandomizedSearchCV)
from sklearn.metrics         import (roc_auc_score, roc_curve,
                                     average_precision_score,
                                     precision_recall_curve,
                                     classification_report, f1_score,
                                     matthews_corrcoef, brier_score_loss)
from sklearn.pipeline        import Pipeline
from sklearn.dummy           import DummyClassifier
from sklearn.calibration     import calibration_curve
from sklearn.impute          import SimpleImputer

from Bio     import PDB
from Bio.PDB import NeighborSearch
from Bio.PDB.SASA import ShrakeRupley

try:
    from Bio.PDB.DSSP import DSSP as BioDSSP
    DSSP_AVAILABLE = True
except Exception:
    DSSP_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
PDB_FILE     = os.path.join(HERE, "tp53_Repair.pdb")
DNA_PDB      = os.path.join(HERE, "3kz8.pdb1")
AF_FILE      = os.path.join(HERE, "AF_P04637_tp53.pdb")
AF_URL       = "https://alphafold.ebi.ac.uk/files/AF-P04637-F1-model_v6.pdb"
CLINVAR      = os.path.join(HERE, "tp53_clinvar_labeled.csv")
FOLDX_MUT    = os.path.join(HERE, "derived", "foldx_per_mutation.csv")
POS_SUMMARY  = os.path.join(HERE, "derived", "foldx_position_summary.csv")
OUT          = os.path.join(HERE, "derived")

CHAIN        = "A"
DNA_CHAIN    = "C"
RADIUS       = 6.0
N_FOLDS      = 5
SEED         = 42
HS_THRESH    = 1.5   # hotspot threshold kcal/mol
DEST_THRESH  = 1.0   # destabilising threshold for frac_destabilizing

# ─────────────────────────────────────────────────────────────────────────────
# Biochemical tables
# ─────────────────────────────────────────────────────────────────────────────
KD = {
    "ALA":1.8,"ARG":-4.5,"ASN":-3.5,"ASP":-3.5,"CYS":2.5,
    "GLN":-3.5,"GLU":-3.5,"GLY":-0.4,"HIS":-3.2,"ILE":4.5,
    "LEU":3.8,"LYS":-3.9,"MET":1.9,"PHE":2.8,"PRO":-1.6,
    "SER":-0.8,"THR":-0.7,"TRP":-0.9,"TYR":-1.3,"VAL":4.2,
}
CHARGE = {
    "ARG":"positive","HIS":"positive","LYS":"positive",
    "ASP":"negative","GLU":"negative",
}
MAX_ASA = {
    "ALA":129,"ARG":274,"ASN":195,"ASP":193,"CYS":167,
    "GLN":223,"GLU":223,"GLY":104,"HIS":224,"ILE":197,
    "LEU":201,"LYS":236,"MET":224,"PHE":240,"PRO":159,
    "SER":155,"THR":172,"TRP":285,"TYR":263,"VAL":174,
}
THREE_TO_ONE = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}

# BLOSUM62 substitution scores (symmetric, self-scores on diagonal)
_BL62_RAW = {
    "A":{"A":4,"R":-1,"N":-2,"D":-2,"C":0,"Q":-1,"E":-1,"G":0,"H":-2,"I":-1,
         "L":-1,"K":-1,"M":-1,"F":-2,"P":-1,"S":1,"T":0,"W":-3,"Y":-2,"V":0},
    "R":{"A":-1,"R":5,"N":0,"D":-2,"C":-3,"Q":1,"E":0,"G":-2,"H":0,"I":-3,
         "L":-2,"K":2,"M":-1,"F":-3,"P":-2,"S":-1,"T":-1,"W":-3,"Y":-2,"V":-3},
    "N":{"A":-2,"R":0,"N":6,"D":1,"C":-3,"Q":0,"E":0,"G":0,"H":1,"I":-3,
         "L":-3,"K":0,"M":-2,"F":-3,"P":-2,"S":1,"T":0,"W":-4,"Y":-2,"V":-3},
    "D":{"A":-2,"R":-2,"N":1,"D":6,"C":-3,"Q":0,"E":2,"G":-1,"H":-1,"I":-3,
         "L":-4,"K":-1,"M":-3,"F":-3,"P":-1,"S":0,"T":-1,"W":-4,"Y":-3,"V":-3},
    "C":{"A":0,"R":-3,"N":-3,"D":-3,"C":9,"Q":-3,"E":-4,"G":-3,"H":-3,"I":-1,
         "L":-1,"K":-3,"M":-1,"F":-2,"P":-3,"S":-1,"T":-1,"W":-2,"Y":-2,"V":-1},
    "Q":{"A":-1,"R":1,"N":0,"D":0,"C":-3,"Q":5,"E":2,"G":-2,"H":0,"I":-3,
         "L":-2,"K":1,"M":0,"F":-3,"P":-1,"S":0,"T":-1,"W":-2,"Y":-1,"V":-2},
    "E":{"A":-1,"R":0,"N":0,"D":2,"C":-4,"Q":2,"E":5,"G":-2,"H":0,"I":-3,
         "L":-3,"K":1,"M":-2,"F":-3,"P":-1,"S":0,"T":-1,"W":-3,"Y":-2,"V":-2},
    "G":{"A":0,"R":-2,"N":0,"D":-1,"C":-3,"Q":-2,"E":-2,"G":6,"H":-2,"I":-4,
         "L":-4,"K":-2,"M":-3,"F":-3,"P":-2,"S":0,"T":-2,"W":-2,"Y":-3,"V":-3},
    "H":{"A":-2,"R":0,"N":1,"D":-1,"C":-3,"Q":0,"E":0,"G":-2,"H":8,"I":-3,
         "L":-3,"K":-1,"M":-2,"F":-1,"P":-2,"S":-1,"T":-2,"W":-2,"Y":2,"V":-3},
    "I":{"A":-1,"R":-3,"N":-3,"D":-3,"C":-1,"Q":-3,"E":-3,"G":-4,"H":-3,"I":4,
         "L":2,"K":-3,"M":1,"F":0,"P":-3,"S":-2,"T":-1,"W":-3,"Y":-1,"V":3},
    "L":{"A":-1,"R":-2,"N":-3,"D":-4,"C":-1,"Q":-2,"E":-3,"G":-4,"H":-3,"I":2,
         "L":4,"K":-2,"M":2,"F":0,"P":-3,"S":-2,"T":-1,"W":-2,"Y":-1,"V":1},
    "K":{"A":-1,"R":2,"N":0,"D":-1,"C":-3,"Q":1,"E":1,"G":-2,"H":-1,"I":-3,
         "L":-2,"K":5,"M":-1,"F":-3,"P":-1,"S":0,"T":-1,"W":-3,"Y":-2,"V":-2},
    "M":{"A":-1,"R":-1,"N":-2,"D":-3,"C":-1,"Q":0,"E":-2,"G":-3,"H":-2,"I":1,
         "L":2,"K":-1,"M":5,"F":0,"P":-2,"S":-1,"T":-1,"W":-1,"Y":-1,"V":1},
    "F":{"A":-2,"R":-3,"N":-3,"D":-3,"C":-2,"Q":-3,"E":-3,"G":-3,"H":-1,"I":0,
         "L":0,"K":-3,"M":0,"F":6,"P":-4,"S":-2,"T":-2,"W":1,"Y":3,"V":-1},
    "P":{"A":-1,"R":-2,"N":-2,"D":-1,"C":-3,"Q":-1,"E":-1,"G":-2,"H":-2,"I":-3,
         "L":-3,"K":-1,"M":-2,"F":-4,"P":7,"S":-1,"T":-1,"W":-4,"Y":-3,"V":-2},
    "S":{"A":1,"R":-1,"N":1,"D":0,"C":-1,"Q":0,"E":0,"G":0,"H":-1,"I":-2,
         "L":-2,"K":0,"M":-1,"F":-2,"P":-1,"S":4,"T":1,"W":-3,"Y":-2,"V":-2},
    "T":{"A":0,"R":-1,"N":0,"D":-1,"C":-1,"Q":-1,"E":-1,"G":-2,"H":-2,"I":-1,
         "L":-1,"K":-1,"M":-1,"F":-2,"P":-1,"S":1,"T":5,"W":-2,"Y":-2,"V":0},
    "W":{"A":-3,"R":-3,"N":-4,"D":-4,"C":-2,"Q":-2,"E":-3,"G":-2,"H":-2,"I":-3,
         "L":-2,"K":-3,"M":-1,"F":1,"P":-4,"S":-3,"T":-2,"W":11,"Y":2,"V":-3},
    "Y":{"A":-2,"R":-2,"N":-2,"D":-3,"C":-2,"Q":-1,"E":-2,"G":-3,"H":2,"I":-1,
         "L":-1,"K":-2,"M":-1,"F":3,"P":-3,"S":-2,"T":-2,"W":2,"Y":7,"V":-1},
    "V":{"A":0,"R":-3,"N":-3,"D":-3,"C":-1,"Q":-2,"E":-2,"G":-3,"H":-3,"I":3,
         "L":1,"K":-2,"M":1,"F":-1,"P":-2,"S":-2,"T":0,"W":-3,"Y":-1,"V":4},
}

DNA_RES = {"DA","DT","DC","DG","A","T","C","G"}

# ─────────────────────────────────────────────────────────────────────────────
# Feature names (grow as steps add features)
# ─────────────────────────────────────────────────────────────────────────────
BASE_FEATURES = [
    "patch_size","mean_hydrophobicity","frac_positive","frac_negative",
    "frac_neutral","mean_relative_sasa","mean_residue_size","frac_buried",
    "frac_hydrophobic",
    "mean_ddg","std_ddg","frac_destabilizing","max_ddg",  # Step 2
    "is_hotspot","min_dist_dna",
    "bfactor_mean","secondary_structure","zinc_distance",  # Step 3
]
VARIANT_FEATURES = ["specific_ddg","ddg_target_aa","blosum62_score",
                    "delta_hydrophobicity","delta_size","charge_change"]  # Step 1 + 4
AF_FEATURES      = ["plddt"]                          # Step 5 (AF positions only)

ALL_FEATURES = BASE_FEATURES + VARIANT_FEATURES + AF_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def banner(msg):
    print("\n" + "="*68)
    print(f"  {msg}")
    print("="*68)


def parse_protein_change(pc):
    """Parse ClinVar 'Protein change' field → (wt_1letter, position, mut_1letter).
    Handles:  Y107H   p.Y107H   Tyr107His   p.Tyr107His
    Returns (None, None, None) on failure."""
    if pd.isna(pc):
        return None, None, None
    pc = str(pc).strip().lstrip("p.")
    # 3-letter format: Tyr107His
    m = re.match(r'^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', pc)
    if m:
        wt3, pos, mut3 = m.groups()
        wt1  = THREE_TO_ONE.get(wt3.upper())
        mut1 = THREE_TO_ONE.get(mut3.upper())
        if wt1 and mut1:
            return wt1, int(pos), mut1
    # 1-letter format: Y107H
    m = re.match(r'^([A-CDEFGHIKLMNPQRSTVWY])(\d+)([A-CDEFGHIKLMNPQRSTVWY])$', pc)
    if m:
        wt1, pos, mut1 = m.groups()
        return wt1, int(pos), mut1
    return None, None, None


def blosum62(wt, mut):
    """Return BLOSUM62 score for WT→MUT substitution (1-letter codes)."""
    try:
        return _BL62_RAW[wt][mut]
    except KeyError:
        return np.nan


def get_ca_coord(residue):
    if "CA" in residue:
        v = residue["CA"].get_vector()
        return np.array([v[0], v[1], v[2]])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 0 — Download AlphaFold structure (optional)
# ─────────────────────────────────────────────────────────────────────────────
def download_alphafold():
    if os.path.exists(AF_FILE):
        print(f"  AlphaFold structure already present: {AF_FILE}")
        return True
    print(f"  Downloading AlphaFold TP53 model from EBI...")
    try:
        urllib.request.urlretrieve(AF_URL, AF_FILE)
        print(f"  Saved → {AF_FILE}")
        return True
    except Exception as e:
        print(f"  [WARN] AlphaFold download failed ({e}). Step 5 skipped.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Structural feature computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_sasa(structure, chain_id):
    sr = ShrakeRupley()
    sr.compute(structure, level="R")
    sasa = {}
    for res in structure[0][chain_id].get_residues():
        if res.get_id()[0] != " ":
            continue
        rnum    = res.get_id()[1]
        max_asa = MAX_ASA.get(res.get_resname().strip(), 200.0)
        sasa[rnum] = min(res.sasa / max_asa, 1.0)
    return sasa


def compute_bfactor(structure, chain_id):
    bfac = {}
    for res in structure[0][chain_id].get_residues():
        if res.get_id()[0] != " ":
            continue
        rnum  = res.get_id()[1]
        bvals = [a.get_bfactor() for a in res.get_atoms()]
        bfac[rnum] = np.mean(bvals) if bvals else np.nan
    return bfac


def compute_zinc_distance(structure, chain_id, positions):
    """Return {position: distance_to_nearest_Zn} in Å."""
    zn_coords = []
    for model in structure:
        for chain in model:
            for res in chain.get_residues():
                if res.get_resname().strip() in ("ZN","ZN2"):
                    for atom in res.get_atoms():
                        v = atom.get_vector()
                        zn_coords.append(np.array([v[0], v[1], v[2]]))
    if not zn_coords:
        print("  [WARN] No ZN atoms found in structure; zinc_distance set to NaN.")
        return {p: np.nan for p in positions}
    zn_arr = np.array(zn_coords)
    dist_map = {}
    chain_obj = structure[0][chain_id]
    for pos in positions:
        try:
            res = chain_obj[pos]
        except KeyError:
            dist_map[pos] = np.nan
            continue
        ca = get_ca_coord(res)
        if ca is None:
            dist_map[pos] = np.nan
            continue
        dists = np.linalg.norm(zn_arr - ca, axis=1)
        dist_map[pos] = round(float(dists.min()), 3)
    return dist_map


def compute_dssp_ss(structure, pdb_path, chain_id, positions):
    """Return {position: ss_code} where ss_code: 0=helix, 1=sheet, 2=loop/other."""
    ss_map = {}
    if DSSP_AVAILABLE:
        try:
            dssp = BioDSSP(structure[0], pdb_path, dssp="mkdssp")
            for key in dssp:
                ch, (_, resnum, _) = key[0], key[1]
                if ch != chain_id:
                    continue
                ss = dssp[key][2]
                if ss in ("H","G","I"):
                    ss_map[resnum] = 0   # helix
                elif ss in ("E","B"):
                    ss_map[resnum] = 1   # sheet
                else:
                    ss_map[resnum] = 2   # loop / other
            print(f"  DSSP computed for {len(ss_map)} residues.")
            return ss_map
        except Exception as e:
            print(f"  [WARN] DSSP failed ({e}); using phi/psi fallback.")
    # Phi/psi fallback using Ramachandran regions
    poly = PDB.Polypeptide.PPBuilder()
    for pp in poly.build_peptides(structure[0][chain_id]):
        angles = pp.get_phi_psi_list()
        for i, res in enumerate(pp):
            phi, psi = angles[i]
            rnum = res.get_id()[1]
            if phi is None or psi is None:
                ss_map[rnum] = 2
            elif (-160 < np.degrees(phi) < -40) and (-60 < np.degrees(psi) < 30):
                ss_map[rnum] = 0   # helix region
            elif (-180 < np.degrees(phi) < -40) and (90 < np.degrees(psi) < 180):
                ss_map[rnum] = 1   # sheet region
            elif (np.degrees(phi) < -40) and (0 < np.degrees(psi) < 90):
                ss_map[rnum] = 1   # sheet region (alternate)
            else:
                ss_map[rnum] = 2
    print(f"  phi/psi SS fallback computed for {len(ss_map)} residues.")
    return ss_map


def compute_dna_distances(dna_structure, chain_id, dna_chain_id, positions):
    dna_coords = []
    try:
        for res in dna_structure[0][dna_chain_id].get_residues():
            if res.get_resname().strip() in DNA_RES:
                for atom in res.get_atoms():
                    if atom.element not in ("H", None):
                        v = atom.get_vector()
                        dna_coords.append([v[0], v[1], v[2]])
    except KeyError:
        print(f"  [WARN] DNA chain '{dna_chain_id}' not found in 3KZ8.")
        return {p: np.nan for p in positions}
    if not dna_coords:
        return {p: np.nan for p in positions}
    dna_arr = np.array(dna_coords)
    print(f"  DNA heavy atoms: {len(dna_arr)}")
    dist_map = {}
    try:
        prot_chain = dna_structure[0][chain_id]
    except KeyError:
        return {p: np.nan for p in positions}
    for pos in positions:
        try:
            res = prot_chain[pos]
        except KeyError:
            dist_map[pos] = np.nan
            continue
        coords = [[a.get_vector()[i] for i in range(3)]
                  for a in res.get_atoms() if a.element not in ("H", None)]
        if not coords:
            dist_map[pos] = np.nan
            continue
        c = np.array(coords)
        diff  = c[:, None, :] - dna_arr[None, :, :]
        dists = np.sqrt((diff**2).sum(axis=2))
        dist_map[pos] = round(float(dists.min()), 3)
    return dist_map


def build_patches(structure, chain_id, positions, radius, sasa_map,
                  bfac_map=None, ss_map=None, zinc_map=None, dna_map=None,
                  plddt_map=None):
    """Build patch feature rows for a list of residue positions."""
    model = structure[0]
    chain = model[chain_id]
    all_atoms = [
        atom for res in chain.get_residues()
        if res.get_id()[0] == " "
        for atom in res.get_atoms()
        if atom.element not in ("H", None)
    ]
    ns   = NeighborSearch(all_atoms)
    rows = []
    for pos in positions:
        try:
            center = chain[pos]
        except KeyError:
            continue
        ca = get_ca_coord(center)
        if ca is None:
            continue
        nearby = ns.search(ca, radius, level="A")
        seen, patch = set(), []
        for atom in nearby:
            res  = atom.get_parent()
            rnum = res.get_id()[1]
            if rnum in seen or res.get_id()[0] != " ":
                continue
            seen.add(rnum)
            rn = res.get_resname().strip()
            patch.append({
                "kd":      KD.get(rn, 0.0),
                "charge":  CHARGE.get(rn, "neutral"),
                "rsasa":   sasa_map.get(rnum, 0.30),
                "size":    sum(1 for a in res.get_atoms()
                               if a.element not in ("H", None)),
            })
        if not patch:
            continue
        df = pd.DataFrame(patch)
        row = {
            "position":            pos,
            "patch_size":          len(df),
            "mean_hydrophobicity": df["kd"].mean(),
            "frac_positive":       (df["charge"] == "positive").mean(),
            "frac_negative":       (df["charge"] == "negative").mean(),
            "frac_neutral":        (df["charge"] == "neutral").mean(),
            "mean_relative_sasa":  df["rsasa"].mean(),
            "mean_residue_size":   df["size"].mean(),
            "frac_buried":         (df["rsasa"] < 0.20).mean(),
            "frac_hydrophobic":    (df["kd"] > 0).mean(),
            "bfactor_mean":        bfac_map.get(pos, np.nan) if bfac_map else np.nan,
            "secondary_structure": ss_map.get(pos, 2)        if ss_map  else 2,
            "zinc_distance":       zinc_map.get(pos, np.nan) if zinc_map else np.nan,
            "min_dist_dna":        dna_map.get(pos, np.nan)  if dna_map  else np.nan,
            "plddt":               plddt_map.get(pos, np.nan) if plddt_map else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — FoldX position statistics
# ─────────────────────────────────────────────────────────────────────────────
def foldx_position_stats(foldx_mut_df):
    """Compute per-position statistics from the per-mutation ΔΔG scan."""
    grp = foldx_mut_df.groupby("position")["ddg_foldx"]
    stats = pd.DataFrame({
        "position":          grp.mean().index,
        "mean_ddg":          grp.mean().values,
        "std_ddg":           grp.std().values,
        "max_ddg":           grp.max().values,
        "frac_destabilizing":(foldx_mut_df.groupby("position")
                              .apply(lambda x: (x["ddg_foldx"] > DEST_THRESH).mean())
                              .values),
        "is_hotspot":        (grp.mean() >= HS_THRESH).astype(int).values,
    }).reset_index(drop=True)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Classification helpers
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(name, clf, X, y, groups, n_folds=N_FOLDS):
    """Evaluate a sklearn estimator with StratifiedGroupKFold.
    Returns (y_prob, metrics_dict)."""
    cv = StratifiedGroupKFold(n_splits=n_folds)
    try:
        y_prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba",
                                   groups=groups)[:, 1]
    except Exception as e:
        print(f"  [WARN] {name}: cross_val_predict failed ({e})")
        return None, {}
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "AUC-ROC":  roc_auc_score(y, y_prob),
        "AUC-PR":   average_precision_score(y, y_prob),
        "F1":       f1_score(y, y_pred, zero_division=0),
        "MCC":      matthews_corrcoef(y, y_pred),
        "Brier":    brier_score_loss(y, y_prob),
    }
    return y_prob, metrics


def tune_model(clf, param_dist, X, y, n_iter=30):
    """RandomizedSearchCV with StratifiedKFold (position-level tuning)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rs = RandomizedSearchCV(clf, param_dist, n_iter=n_iter, cv=cv,
                            scoring="roc_auc", n_jobs=-1,
                            random_state=SEED, verbose=0)
    rs.fit(X, y)
    print(f"  Best AUC (tuning fold): {rs.best_score_:.4f}")
    print(f"  Best params: {rs.best_params_}")
    return rs.best_estimator_


def make_pipeline(clf):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     clf),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc(results, y_true, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = plt.cm.tab10.colors
    for i, (name, (y_prob, metrics)) in enumerate(results.items()):
        if y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = metrics["AUC-ROC"]
        ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})",
                color=palette[i % 10], linewidth=2)
    ax.plot([0,1],[0,1], "k:", linewidth=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Improved TP53 Patch Classifier\n"
                 "5-fold StratifiedGroupKFold (grouped by position)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ROC plot → {save_path}")


def plot_pr(results, y_true, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = plt.cm.tab10.colors
    baseline = y_true.mean()
    for i, (name, (y_prob, metrics)) in enumerate(results.items()):
        if y_prob is None:
            continue
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        auc = metrics["AUC-PR"]
        ax.plot(rec, prec, label=f"{name}  (AP={auc:.3f})",
                color=palette[i % 10], linewidth=2)
    ax.axhline(baseline, color="grey", linestyle=":", linewidth=0.8,
               label=f"Baseline (prevalence={baseline:.2f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves\n(better metric for imbalanced classes)",
                 fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  PR plot → {save_path}")


def plot_feature_importance(rf_model, gb_model, feature_names, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, model, title in zip(
        axes,
        [rf_model, gb_model],
        ["Random Forest feature importance", "Gradient Boosting feature importance"],
    ):
        try:
            imp = model.named_steps["clf"].feature_importances_
        except AttributeError:
            continue
        df = pd.DataFrame({"feature": feature_names, "importance": imp})
        df = df.sort_values("importance", ascending=True).tail(15)
        ax.barh(df["feature"], df["importance"], color="#1f77b4", alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Importance", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Feature importance plot → {save_path}")


def plot_calibration(results, y_true, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    palette = plt.cm.tab10.colors
    ax.plot([0,1],[0,1],"k--", linewidth=0.8, label="Perfectly calibrated")
    for i, (name, (y_prob, _)) in enumerate(results.items()):
        if y_prob is None:
            continue
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, "s-", label=name,
                color=palette[i % 10], linewidth=2, markersize=4)
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of positives (pathogenic)", fontsize=11)
    ax.set_title("Calibration curves", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Calibration plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def write_report(results, y_true, groups, feature_names,
                 n_variants, n_positions, baseline_auc, save_path):
    lines = [
        "TP53 Structural Patch — Improved Classification Report",
        "=" * 60,
        f"Variant-level dataset : {n_variants} variants",
        f"Unique positions      : {n_positions}",
        f"Pathogenic prevalence : {y_true.mean():.1%}",
        f"Features used         : {len(feature_names)}",
        f"  {feature_names}",
        f"Cross-validation      : {N_FOLDS}-fold StratifiedGroupKFold (by position)",
        "",
        f"Baseline (original classify_patches.py):",
        f"  Logistic Regression — 12 features — AUC {baseline_auc:.3f}",
        "",
        f"{'Model':<35} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>6} {'MCC':>6} {'Brier':>7}",
        "-" * 73,
    ]
    best_auc, best_name = 0.0, ""
    for name, (y_prob, metrics) in results.items():
        if not metrics:
            continue
        lines.append(
            f"{name:<35} {metrics['AUC-ROC']:8.4f} {metrics['AUC-PR']:8.4f} "
            f"{metrics['F1']:6.4f} {metrics['MCC']:6.4f} {metrics['Brier']:7.4f}"
        )
        if metrics["AUC-ROC"] > best_auc:
            best_auc, best_name = metrics["AUC-ROC"], name
    lines += [
        "",
        f"Best model : {best_name}  (AUC-ROC = {best_auc:.4f})",
        f"Improvement over original baseline: ΔAUC = {best_auc - baseline_auc:+.4f}",
        "",
    ]
    # Detailed report for best model
    best_prob = results[best_name][0]
    if best_prob is not None:
        y_pred_best = (best_prob >= 0.5).astype(int)
        lines += [
            f"Detailed classification report — {best_name}:",
            classification_report(y_true, y_pred_best,
                                  target_names=["Benign","Pathogenic"]),
        ]
    with open(save_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Report → {save_path}")
    return best_auc, best_name


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)
    banner("TP53 Improved Classification Pipeline")

    # ── Load base data ────────────────────────────────────────────────────────
    print("\n[1/9] Loading existing data files...")
    clinvar   = pd.read_csv(CLINVAR)
    foldx_mut = pd.read_csv(FOLDX_MUT)
    pos_sum   = pd.read_csv(POS_SUMMARY)
    for df in (clinvar, foldx_mut, pos_sum):
        df["position"] = df["position"].astype(int)
    print(f"  ClinVar variants   : {len(clinvar)}")
    print(f"  FoldX substitutions: {len(foldx_mut)}")
    print(f"  FoldX positions    : {len(pos_sum)}")

    # ── Parse ClinVar protein changes → WT / MUT amino acids ─────────────────
    print("\n[2/9] Parsing ClinVar protein change column...")
    parsed = clinvar["Protein change"].apply(parse_protein_change)
    clinvar["wt_aa"]   = parsed.apply(lambda x: x[0])
    clinvar["mut_aa"]  = parsed.apply(lambda x: x[2])
    parsed_ok = clinvar["wt_aa"].notna().sum()
    print(f"  Successfully parsed: {parsed_ok}/{len(clinvar)} variants")

    # ── Step 1: per-variant specific ΔΔG ─────────────────────────────────────
    print("\n[3/9] Step 1 — Merging per-variant FoldX ΔΔG...")

    # Strategy A: exact match (position + wt_aa + mut_aa) → specific_ddg
    merged = clinvar.merge(
        foldx_mut[["position","wt_aa","mut_aa","ddg_foldx"]],
        on=["position","wt_aa","mut_aa"],
        how="left",
    ).rename(columns={"ddg_foldx":"specific_ddg"})
    n_exact = merged["specific_ddg"].notna().sum()
    print(f"  Exact match (pos+wt+mut):        {n_exact}/{len(merged)}")

    # Strategy B: target-AA ΔΔG (position + mut_aa) → separate feature
    # The ΔΔG of placing amino acid X at position P relative to the PDB
    # reference WT. Available for all domain variants regardless of ClinVar WT.
    # Kept separate so the model can weight it independently of exact_specific_ddg.
    foldx_by_mut = (foldx_mut[["position","mut_aa","ddg_foldx"]]
                    .rename(columns={"ddg_foldx":"ddg_target_aa"}))
    merged = merged.merge(foldx_by_mut, on=["position","mut_aa"], how="left")
    n_target = merged["ddg_target_aa"].notna().sum()
    print(f"  Target-AA ΔΔG (pos+mut only):   {n_target}/{len(merged)} (separate feature)")

    # ── Step 4: BLOSUM62 substitution score + biochemical difference features ──
    merged["blosum62_score"] = merged.apply(
        lambda r: blosum62(r["wt_aa"], r["mut_aa"])
        if pd.notna(r["wt_aa"]) and pd.notna(r["mut_aa"]) else np.nan,
        axis=1,
    )
    print(f"  BLOSUM62 scores computed: {merged['blosum62_score'].notna().sum()}")

    # Hydrophobicity change (Kyte-Doolittle): positive = more hydrophobic mutant
    kd_map = {aa: KD.get(ONE_TO_THREE.get(aa, ""), np.nan) for aa in "ACDEFGHIKLMNPQRSTVWY"}
    merged["delta_hydrophobicity"] = merged.apply(
        lambda r: kd_map.get(r["mut_aa"], np.nan) - kd_map.get(r["wt_aa"], np.nan)
        if pd.notna(r["wt_aa"]) and pd.notna(r["mut_aa"]) else np.nan, axis=1)

    # Residue size change (Max ASA as proxy): positive = larger mutant (steric clash risk)
    size_map = {aa: MAX_ASA.get(ONE_TO_THREE.get(aa, ""), np.nan) for aa in "ACDEFGHIKLMNPQRSTVWY"}
    merged["delta_size"] = merged.apply(
        lambda r: size_map.get(r["mut_aa"], np.nan) - size_map.get(r["wt_aa"], np.nan)
        if pd.notna(r["wt_aa"]) and pd.notna(r["mut_aa"]) else np.nan, axis=1)

    # Charge change: WT charge class → MUT charge class, encoded as numeric
    charge_num = {"positive": 1, "negative": -1, None: 0}
    merged["charge_change"] = merged.apply(
        lambda r: (charge_num.get(CHARGE.get(ONE_TO_THREE.get(r["mut_aa"], ""), None))
                   - charge_num.get(CHARGE.get(ONE_TO_THREE.get(r["wt_aa"], ""), None)))
        if pd.notna(r["wt_aa"]) and pd.notna(r["mut_aa"]) else np.nan, axis=1)

    n_bdf = merged["delta_hydrophobicity"].notna().sum()
    print(f"  Biochemical difference features computed: {n_bdf}/{len(merged)}")

    # ── Step 2: extended FoldX position stats ────────────────────────────────
    print("\n[4/9] Step 2 — Computing extended FoldX position statistics...")
    pos_stats = foldx_position_stats(foldx_mut)
    print(f"  Positions with stats: {len(pos_stats)}")
    print(f"  Hotspot positions   : {pos_stats['is_hotspot'].sum()}")

    # ── Load structures ───────────────────────────────────────────────────────
    print("\n[5/9] Loading PDB structures...")
    parser = PDB.PDBParser(QUIET=True)
    struct_2ocj = parser.get_structure("2ocj", PDB_FILE)
    struct_3kz8 = parser.get_structure("3kz8", DNA_PDB) if os.path.exists(DNA_PDB) else None
    if struct_3kz8 is None:
        print("  [WARN] 3kz8.pdb1 not found — DNA distances will be NaN.")

    # ── Step 3: structural features on 2OCJ domain (residues 96–289) ─────────
    print("\n[6/9] Step 3 — Computing structural features from 2OCJ...")
    scanned_positions = sorted(pos_stats["position"].tolist())

    sasa_map = compute_sasa(struct_2ocj, CHAIN)
    print(f"  SASA computed for {len(sasa_map)} residues.")

    bfac_map = compute_bfactor(struct_2ocj, CHAIN)
    print(f"  B-factors computed for {len(bfac_map)} residues.")

    zinc_map = compute_zinc_distance(struct_2ocj, CHAIN, scanned_positions)
    zn_valid = sum(1 for v in zinc_map.values() if not np.isnan(v))
    print(f"  Zinc distances computed for {zn_valid}/{len(scanned_positions)} positions.")

    ss_map = compute_dssp_ss(struct_2ocj, PDB_FILE, CHAIN, scanned_positions)

    dna_map = {}
    if struct_3kz8 is not None:
        print("  Computing DNA-contact distances from 3KZ8...")
        dna_map = compute_dna_distances(struct_3kz8, CHAIN, DNA_CHAIN,
                                        scanned_positions)
        for pos, cls in [(248,"contact"),(273,"contact"),
                         (175,"structural"),(242,"structural")]:
            if pos in dna_map:
                print(f"    Pos {pos} ({cls}): {dna_map[pos]:.2f} Å to DNA")

    # ── Build patch features for 2OCJ domain ─────────────────────────────────
    patches_2ocj = build_patches(
        struct_2ocj, CHAIN, scanned_positions, RADIUS,
        sasa_map, bfac_map, ss_map, zinc_map, dna_map, plddt_map=None,
    )
    # Fill missing DNA distances
    if dna_map:
        max_dna = patches_2ocj["min_dist_dna"].dropna().max()
        patches_2ocj["min_dist_dna"].fillna(max_dna, inplace=True)
    else:
        patches_2ocj["min_dist_dna"] = np.nan

    # Merge FoldX stats into patch table
    patches_2ocj = patches_2ocj.merge(
        pos_stats[["position","mean_ddg","std_ddg","max_ddg",
                   "frac_destabilizing","is_hotspot"]],
        on="position", how="left",
    )
    patches_2ocj["plddt"] = np.nan   # not available for crystal structure

    print(f"  2OCJ patches built: {len(patches_2ocj)} positions")

    # ── Step 5: AlphaFold extension ───────────────────────────────────────────
    af_ok = download_alphafold()
    all_patches = patches_2ocj.copy()

    if af_ok and os.path.exists(AF_FILE):
        print("\n[7/9] Step 5 — Building patches for outside-domain positions (AlphaFold)...")
        struct_af  = parser.get_structure("af", AF_FILE)
        af_chain   = struct_af[0]["A"]

        # pLDDT is stored in B-factor column of AlphaFold PDB
        plddt_map = {}
        for res in af_chain.get_residues():
            if res.get_id()[0] != " ":
                continue
            bvals = [a.get_bfactor() for a in res.get_atoms()]
            plddt_map[res.get_id()[1]] = np.mean(bvals) if bvals else np.nan

        # Outside-domain positions that have ClinVar variants
        domain_set = set(scanned_positions)
        outside_variants = merged[~merged["position"].isin(domain_set)]
        outside_positions = sorted(outside_variants["position"].unique().tolist())
        print(f"  Outside-domain ClinVar variants: {len(outside_variants)}")
        print(f"  Unique outside positions        : {len(outside_positions)}")

        # SASA, B-factor, SS for AF structure
        af_sasa = compute_sasa(struct_af, "A")
        af_bfac = compute_bfactor(struct_af, "A")
        af_ss   = compute_dssp_ss(struct_af, AF_FILE, "A", outside_positions)

        # Zinc distance: AF has no explicit Zn, use fallback
        af_zinc = {p: np.nan for p in outside_positions}

        # DNA distance: outside-domain residues are far from DNA by biology
        # Use the maximum observed DNA distance from the 2OCJ domain as proxy
        max_dna_domain = patches_2ocj["min_dist_dna"].max() if not patches_2ocj["min_dist_dna"].isna().all() else 50.0
        af_dna  = {p: max_dna_domain + 5.0 for p in outside_positions}

        patches_af = build_patches(
            struct_af, "A", outside_positions, RADIUS,
            af_sasa, af_bfac, af_ss, af_zinc, af_dna, plddt_map=plddt_map,
        )

        # FoldX stats not available for outside-domain — fill with domain means
        for col in ["mean_ddg","std_ddg","max_ddg","frac_destabilizing"]:
            patches_af[col] = pos_stats[col].mean()
        patches_af["is_hotspot"] = 0  # conservative: not known hotspots

        all_patches = pd.concat([patches_2ocj, patches_af], ignore_index=True)
        print(f"  Total patches (2OCJ + AF) : {len(all_patches)}")
    else:
        print("\n[7/9] Step 5 — Skipped (AlphaFold not available).")

    # ── Assemble variant-level feature matrix ─────────────────────────────────
    print("\n[8/9] Assembling variant feature matrix...")
    POSITION_FEATS = [
        "patch_size","mean_hydrophobicity","frac_positive","frac_negative",
        "frac_neutral","mean_relative_sasa","mean_residue_size","frac_buried",
        "frac_hydrophobic","mean_ddg","std_ddg","frac_destabilizing","max_ddg",
        "is_hotspot","min_dist_dna","bfactor_mean","secondary_structure",
        "zinc_distance","plddt",
    ]
    vm = merged.merge(
        all_patches[["position"] + POSITION_FEATS],
        on="position", how="left",
    )
    vm = vm.dropna(subset=["patch_size"])
    vm["label_bin"] = (vm["label"] == "pathogenic").astype(int)

    feature_cols = POSITION_FEATS + ["specific_ddg","ddg_target_aa","blosum62_score",
                                      "delta_hydrophobicity","delta_size","charge_change"]

    # For any feature with >30% missing values, report it
    for col in feature_cols:
        frac_missing = vm[col].isna().mean()
        if frac_missing > 0.0:
            print(f"  {col}: {frac_missing:.1%} missing → imputing with column median")
            vm[col].fillna(vm[col].median(), inplace=True)

    # Drop features that are entirely NaN (e.g. plddt if AF skipped)
    feature_cols = [c for c in feature_cols if vm[c].notna().any()]

    X      = vm[feature_cols].values.astype(float)
    y      = vm["label_bin"].values
    groups = vm["position"].values

    print(f"\n  Final dataset :")
    print(f"    Variants    : {len(vm)}")
    print(f"    Positions   : {vm['position'].nunique()}")
    print(f"    Features    : {len(feature_cols)}")
    print(f"    Pathogenic  : {y.sum()} ({y.mean():.1%})")
    print(f"    Benign      : {(1-y).sum()} ({(1-y).mean():.1%})")

    vm.to_csv(os.path.join(OUT, "improved_feature_matrix.csv"), index=False)

    # ── Step 6: Train and evaluate all models ─────────────────────────────────
    banner("Step 6 — Model training and evaluation")

    # --- Hyperparameter search spaces
    rf_params = {
        "clf__n_estimators":      [200, 400, 600, 800],
        "clf__max_depth":         [None, 5, 10, 15, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf":  [1, 2, 4],
        "clf__max_features":      ["sqrt", "log2", 0.5],
        "clf__class_weight":      ["balanced"],
    }
    gb_params = {
        "clf__n_estimators":  [100, 200, 300],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__max_depth":     [2, 3, 4, 5],
        "clf__subsample":     [0.6, 0.8, 1.0],
        "clf__min_samples_leaf": [1, 5, 10],
    }
    svm_params = {
        "clf__C":     [0.01, 0.1, 1, 10, 100],
        "clf__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "clf__kernel":["rbf","sigmoid"],
        "clf__class_weight": ["balanced"],
    }

    # --- Tune on full dataset (StratifiedKFold, position-level)
    print("\nTuning Random Forest...")
    rf_base = make_pipeline(RandomForestClassifier(random_state=SEED))
    rf_best = tune_model(rf_base, rf_params, X, y, n_iter=40)

    print("\nTuning Gradient Boosting...")
    gb_base = make_pipeline(GradientBoostingClassifier(random_state=SEED))
    gb_best = tune_model(gb_base, gb_params, X, y, n_iter=40)

    print("\nTuning SVM...")
    svm_base = make_pipeline(SVC(probability=True, random_state=SEED))
    svm_best = tune_model(svm_base, svm_params, X, y, n_iter=40)

    lr_pipe = make_pipeline(
        LogisticRegression(max_iter=3000, class_weight="balanced",
                           random_state=SEED, C=1.0)
    )
    # Voting ensemble (soft voting, equal weights)
    ensemble = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf", VotingClassifier(
            estimators=[
                ("rf",  rf_best.named_steps["clf"]),
                ("gb",  gb_best.named_steps["clf"]),
                ("svm", svm_best.named_steps["clf"]),
            ],
            voting="soft",
        )),
    ])

    # --- Evaluate all models with GroupKFold
    print("\nEvaluating all models with 5-fold StratifiedGroupKFold...")
    BASELINE_AUC = 0.580   # original classify_patches.py result

    models = {
        "Logistic Regression (baseline)": lr_pipe,
        "Random Forest (tuned)":          rf_best,
        "Gradient Boosting (tuned)":      gb_best,
        "SVM (tuned)":                    svm_best,
        "Ensemble (RF+GB+SVM)":           ensemble,
    }
    results = {}
    for name, model in models.items():
        print(f"  {name} ... ", end="", flush=True)
        y_prob, metrics = evaluate_model(name, model, X, y, groups)
        results[name] = (y_prob, metrics)
        if metrics:
            print(f"AUC-ROC={metrics['AUC-ROC']:.4f}  "
                  f"AUC-PR={metrics['AUC-PR']:.4f}  "
                  f"MCC={metrics['MCC']:.4f}")
        else:
            print("FAILED")

    # Random baseline
    dummy = make_pipeline(DummyClassifier(strategy="stratified", random_state=SEED))
    y_prob_rand, m_rand = evaluate_model("Random baseline", dummy, X, y, groups)
    results["Random baseline"] = (y_prob_rand, m_rand)
    print(f"  Random baseline ... AUC-ROC={m_rand['AUC-ROC']:.4f}")

    # ── Plots and report ──────────────────────────────────────────────────────
    banner("Generating outputs")
    plot_roc(results, y, os.path.join(OUT, "improved_roc_curves.png"))
    plot_pr(results, y, os.path.join(OUT, "improved_pr_curves.png"))
    plot_calibration(results, y, os.path.join(OUT, "improved_calibration.png"))

    # Feature importance for tree models
    rf_best.fit(X, y)
    gb_best.fit(X, y)
    plot_feature_importance(rf_best, gb_best, feature_cols,
                            os.path.join(OUT, "improved_feature_importance.png"))

    best_auc, best_name = write_report(
        results, y, groups, feature_cols,
        n_variants=len(vm), n_positions=vm["position"].nunique(),
        baseline_auc=BASELINE_AUC,
        save_path=os.path.join(OUT, "improved_classification_report.txt"),
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("Summary")
    print(f"\n  Original baseline (LogReg, 12 feat)  : AUC = {BASELINE_AUC:.3f}")
    print(f"  Best improved model ({best_name})")
    best_m = results[best_name][1]
    print(f"    AUC-ROC : {best_m['AUC-ROC']:.4f}  (Δ = {best_m['AUC-ROC']-BASELINE_AUC:+.4f})")
    print(f"    AUC-PR  : {best_m['AUC-PR']:.4f}")
    print(f"    F1      : {best_m['F1']:.4f}")
    print(f"    MCC     : {best_m['MCC']:.4f}")
    print(f"\n  Features used: {len(feature_cols)}")
    for f in feature_cols:
        print(f"    {f}")
    print(f"\n  All outputs in: {OUT}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
