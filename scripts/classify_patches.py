"""
classify_patches.py
===================
Script 6 of 6 — Patch-based variant classification (proof of concept)

THESIS ALIGNMENT:
    Directly addresses the thesis title:
    "Assessing the Validity of 3D Structural Patches for the Classification
    of Point Mutations: Showcase on TP53 Gene"

    And the stated objective:
    "évaluer dans quelle mesure ces descripteurs structuraux locaux permettent
    de discriminer les mutations à effet pathogène de celles à effet neutre"

METHODOLOGICAL NOTES:
    1. DNA-contact distance is computed from 3KZ8 (TP53–DNA complex, chain C
       contains the double-stranded DNA). This adds a 12th feature that captures
       proximity to the DNA interface — necessary to correctly characterise
       contact mutants (R248: 3.6 Å, R273: 10.5 Å) that stability-only
       approaches miss.

    2. Cross-validation uses GroupKFold grouped by residue position, ensuring
       that all variants from the same position appear in either the training
       set or the test set — never both. This avoids pseudo-replication, since
       variants at the same position share identical patch feature vectors.

    3. A position-level analysis (188 positions, binary majority label) is
       performed alongside the variant-level analysis to confirm results are
       not artefacts of repeated identical feature vectors.

INPUT:
    tp53_Repair.pdb
    3kz8.pdb1                            (TP53-DNA complex for contact distances)
    derived/foldx_position_summary.csv
    derived/variants_with_hotspot.csv

OUTPUT:
    derived/all_position_patches.csv
    derived/variant_feature_matrix.csv
    derived/position_feature_matrix.csv
    derived/classification_report.txt
    derived/roc_curve.png
    derived/feature_importance.png
"""

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET SUBSET — 866 VARIANTS (DOMAIN-RESTRICTED, residues 96–289)
# ═══════════════════════════════════════════════════════════════════════════════
# The full ClinVar labeled dataset contains 1,374 variants. This script uses
# only the 866 variants (611 pathogenic, 255 benign) that map to residues
# 96–289 — the region covered by the FoldX PositionScan and the 2OCJ crystal
# structure. The 508 variants at positions outside this range have no patch
# feature vector and are excluded by the dropna(subset=["patch_size"]) call.
#
# This is the BASELINE classifier. The improved classifier in
# tp53_improved_classification.py recovers those 508 variants by computing
# patches from the AlphaFold2 model (AF-P04637) for outside-domain positions,
# and uses a 25-feature vector instead of 12.
#
# PDB CHAIN: all patches are built from chain A of tp53_Repair.pdb.
# DNA distances use chain A (protein) and chain C (DNA) from 3kz8.pdb1.
# ═══════════════════════════════════════════════════════════════════════════════

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

from Bio import PDB
from Bio.PDB import NeighborSearch
from Bio.PDB.SASA import ShrakeRupley

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PDB_FILE    = "tp53_Repair.pdb"
DNA_PDB     = "3kz8.pdb1"
POS_SUMMARY = os.path.join("derived", "foldx_position_summary.csv")
VARIANTS    = os.path.join("derived", "variants_with_hotspot.csv")
OUTPUT_DIR  = "derived"
CHAIN_ID    = "A"
DNA_CHAIN   = "C"
PATCH_RADIUS = 6.0
N_FOLDS     = 5
RANDOM_STATE = 42

KD_HYDROPHOBICITY = {
    "ALA":1.8,"ARG":-4.5,"ASN":-3.5,"ASP":-3.5,"CYS":2.5,
    "GLN":-3.5,"GLU":-3.5,"GLY":-0.4,"HIS":-3.2,"ILE":4.5,
    "LEU":3.8,"LYS":-3.9,"MET":1.9,"PHE":2.8,"PRO":-1.6,
    "SER":-0.8,"THR":-0.7,"TRP":-0.9,"TYR":-1.3,"VAL":4.2,
}
CHARGE = {"ARG":"positive","HIS":"positive","LYS":"positive",
          "ASP":"negative","GLU":"negative"}
MAX_ASA = {
    "ALA":129.0,"ARG":274.0,"ASN":195.0,"ASP":193.0,"CYS":167.0,
    "GLN":223.0,"GLU":223.0,"GLY":104.0,"HIS":224.0,"ILE":197.0,
    "LEU":201.0,"LYS":236.0,"MET":224.0,"PHE":240.0,"PRO":159.0,
    "SER":155.0,"THR":172.0,"TRP":285.0,"TYR":263.0,"VAL":174.0,
}
DNA_RESIDUES = {"DA","DT","DC","DG"}

FEATURES = [
    "patch_size", "mean_hydrophobicity", "frac_positive", "frac_negative",
    "frac_neutral", "mean_relative_sasa", "mean_residue_size",
    "frac_buried", "frac_hydrophobic",
    "mean_ddg",           # FoldX mean ∆∆G — stability signal
    "is_hotspot",         # binary stability threshold
    "min_dist_dna",       # distance to DNA — interface signal
]


# ---------------------------------------------------------------------------
# DNA-contact distance
# ---------------------------------------------------------------------------
def compute_dna_distances(protein_structure, dna_structure,
                          protein_chain_id: str, dna_chain_id: str,
                          positions: list) -> dict:
    """
    For each position, compute the minimum heavy-atom distance
    from that residue to any DNA heavy atom in the complex.
    Uses 3KZ8 chain A (protein) and chain C (DNA).
    Returns dict {position: min_dist_angstrom}.
    """
    # Collect DNA heavy atoms from the DNA chain
    dna_model  = dna_structure[0]
    dna_coords = []
    for res in dna_model[dna_chain_id].get_residues():
        if res.get_resname().strip() in DNA_RESIDUES:
            for atom in res.get_atoms():
                if atom.element != "H" and atom.element is not None:
                    v = atom.get_vector()
                    dna_coords.append(np.array([v[0], v[1], v[2]]))
    dna_coords = np.array(dna_coords)  # shape (N_dna_atoms, 3)
    print(f"  DNA heavy atoms in 3KZ8 chain {dna_chain_id}: {len(dna_coords)}")

    # For each protein residue position, find minimum distance to DNA
    prot_model = dna_structure[0]   # 3KZ8 chain A has same numbering as 2OCJ
    prot_chain = prot_model[protein_chain_id]
    dist_map   = {}

    for pos in positions:
        try:
            res = prot_chain[pos]
        except KeyError:
            dist_map[pos] = np.nan
            continue
        prot_coords = []
        for atom in res.get_atoms():
            if atom.element != "H" and atom.element is not None:
                v = atom.get_vector()
                prot_coords.append(np.array([v[0], v[1], v[2]]))
        if not prot_coords:
            dist_map[pos] = np.nan
            continue
        prot_coords = np.array(prot_coords)
        # Vectorised pairwise distance: shape (n_prot, n_dna) → min
        diff  = prot_coords[:, None, :] - dna_coords[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        dist_map[pos] = round(float(dists.min()), 3)

    return dist_map


# ---------------------------------------------------------------------------
# Patch construction (all positions)
# ---------------------------------------------------------------------------
def build_all_patches(structure, chain_id, positions, radius, sasa_map):
    model = structure[0]
    chain = model[chain_id]
    all_atoms = [
        atom for res in chain.get_residues()
        if res.get_id()[0] == " "
        for atom in res.get_atoms()
        if atom.element != "H" and atom.element is not None
    ]
    ns = NeighborSearch(all_atoms)
    rows = []
    for pos in positions:
        try:
            center_res = chain[pos]
        except KeyError:
            continue
        if "CA" not in center_res:
            continue
        ca = center_res["CA"].get_vector()
        ca_coord = np.array([ca[0], ca[1], ca[2]])
        nearby = ns.search(ca_coord, radius, level="A")
        seen, patch_data = set(), []
        for atom in nearby:
            res  = atom.get_parent()
            rnum = res.get_id()[1]
            if rnum in seen or res.get_id()[0] != " ":
                continue
            seen.add(rnum)
            resname = res.get_resname().strip()
            patch_data.append({
                "hydrophobicity": KD_HYDROPHOBICITY.get(resname, 0.0),
                "charge":         CHARGE.get(resname, "neutral"),
                "relative_sasa":  sasa_map.get(rnum, 0.30),
                "residue_size":   sum(1 for a in res.get_atoms()
                                     if a.element != "H" and a.element is not None),
            })
        if not patch_data:
            continue
        df = pd.DataFrame(patch_data)
        rows.append({
            "position":            pos,
            "patch_size":          len(df),
            "mean_hydrophobicity": df["hydrophobicity"].mean(),
            "frac_positive":       (df["charge"] == "positive").mean(),
            "frac_negative":       (df["charge"] == "negative").mean(),
            "frac_neutral":        (df["charge"] == "neutral").mean(),
            "mean_relative_sasa":  df["relative_sasa"].mean(),
            "mean_residue_size":   df["residue_size"].mean(),
            "frac_buried":         (df["relative_sasa"] < 0.20).mean(),
            "frac_hydrophobic":    (df["hydrophobicity"] > 0).mean(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_classification(X, y, groups, label, n_folds):
    """
    Run logistic regression with GroupKFold (no pseudo-replication).
    groups = position array, ensuring all variants at same position
    stay in the same fold.
    """
    try:
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                  random_state=RANDOM_STATE)
        cv_kwargs = {"groups": groups}
    except TypeError:
        # StratifiedGroupKFold in older sklearn doesn't accept shuffle/random_state
        cv = StratifiedGroupKFold(n_splits=n_folds)
        cv_kwargs = {"groups": groups}

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                      class_weight="balanced")),
    ])
    y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba",
                               **cv_kwargs)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y, y_prob)
    f1  = f1_score(y, y_pred)
    return y_prob, y_pred, auc, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 62)
    print("Patch-Based Classification — Proof of Concept")
    print("Thesis: Assessing the Validity of 3D Structural Patches")
    print("        for the Classification of Point Mutations (TP53)")
    print("=" * 62)

    # ── Load inputs ──────────────────────────────────────────────────────────
    pos_df   = pd.read_csv(POS_SUMMARY)
    pos_df["position"] = pos_df["position"].astype(int)
    all_positions = sorted(pos_df["position"].tolist())

    variants = pd.read_csv(VARIANTS)
    variants["position"] = variants["position"].astype(int)
    print(f"\nPositions to analyse : {len(all_positions)}")
    print(f"Variants loaded      : {len(variants)}  "
          f"(path={( variants.label=='pathogenic').sum()}, "
          f"benign={(variants.label=='benign').sum()})")

    # ── Load structures ───────────────────────────────────────────────────────
    pdb_parser = PDB.PDBParser(QUIET=True)
    struct_2ocj = pdb_parser.get_structure("2ocj", PDB_FILE)

    if not os.path.exists(DNA_PDB):
        print(f"[ERROR] {DNA_PDB} not found — DNA-contact distances cannot be computed.")
        sys.exit(1)
    struct_3kz8 = pdb_parser.get_structure("3kz8", DNA_PDB)
    print(f"\nStructures loaded: {PDB_FILE}, {DNA_PDB}")

    # ── SASA (Shrake-Rupley on repaired structure) ────────────────────────────
    sr = ShrakeRupley()
    sr.compute(struct_2ocj, level="R")
    sasa_map = {}
    for res in struct_2ocj[0][CHAIN_ID].get_residues():
        if res.get_id()[0] != " ":
            continue
        rnum    = res.get_id()[1]
        max_asa = MAX_ASA.get(res.get_resname().strip(), 200.0)
        sasa_map[rnum] = min(res.sasa / max_asa, 1.0)
    print(f"Shrake-Rupley SASA computed for {len(sasa_map)} residues.")

    # ── DNA-contact distances (from 3KZ8, same residue numbering) ────────────
    print("\nComputing DNA-contact distances from 3KZ8...")
    dist_map = compute_dna_distances(
        struct_3kz8, struct_3kz8, CHAIN_ID, DNA_CHAIN, all_positions
    )
    # Sanity check key positions
    for pos, expected_class in [(248, "contact"), (273, "contact"),
                                (175, "structural"), (242, "structural")]:
        if pos in dist_map:
            print(f"  Pos {pos} ({expected_class}): {dist_map[pos]:.2f} A to DNA")

    # ── Build patches (all 194 positions) ────────────────────────────────────
    print(f"\nBuilding patches for {len(all_positions)} positions...")
    patch_df = build_all_patches(struct_2ocj, CHAIN_ID, all_positions,
                                 PATCH_RADIUS, sasa_map)

    # Join FoldX summary + DNA distance
    patch_df = patch_df.merge(
        pos_df[["position", "mean_ddg", "is_hotspot"]], on="position", how="left"
    )
    patch_df["min_dist_dna"] = patch_df["position"].map(dist_map)

    # Fill any missing DNA distances with the domain maximum (far from DNA)
    max_dist = patch_df["min_dist_dna"].max()
    n_missing = patch_df["min_dist_dna"].isna().sum()
    if n_missing:
        patch_df["min_dist_dna"].fillna(max_dist, inplace=True)
        print(f"  {n_missing} positions had no DNA distance → filled with {max_dist:.1f} A")

    all_patch_path = os.path.join(OUTPUT_DIR, "all_position_patches.csv")
    patch_df.to_csv(all_patch_path, index=False)
    print(f"All-position patch table saved → {all_patch_path} ({len(patch_df)} rows)")

    # ── Variant feature matrix ────────────────────────────────────────────────
    print("\nBuilding variant feature matrix...")
    variants_clean = variants.drop(columns=["is_hotspot"], errors="ignore")
    vm = variants_clean.merge(
        patch_df[["position"] + FEATURES], on="position", how="left"
    )
    vm = vm.dropna(subset=["patch_size"])
    vm["label_bin"] = (vm["label"] == "pathogenic").astype(int)

    print(f"Variants with patch features   : {len(vm)}")
    print(f"Variants outside domain dropped: {len(variants)-len(vm)}")
    print(f"Unique positions in analysis   : {vm['position'].nunique()}")
    print(f"Mean variants per position     : {len(vm)/vm['position'].nunique():.1f}")

    feat_path = os.path.join(OUTPUT_DIR, "variant_feature_matrix.csv")
    vm.to_csv(feat_path, index=False)
    print(f"Feature matrix saved → {feat_path}")

    # ── Position-level matrix (one row per position, no pseudo-replication) ───
    print("\nBuilding position-level matrix...")
    pos_variants = (
        vm.groupby("position")
        .agg(
            n_variants=("label_bin", "count"),
            n_pathogenic=("label_bin", "sum"),
            frac_pathogenic=("label_bin", "mean"),
        )
        .reset_index()
    )
    pos_variants["majority_pathogenic"] = (pos_variants["frac_pathogenic"] >= 0.5).astype(int)
    pos_matrix = pos_variants.merge(patch_df[["position"] + FEATURES], on="position")

    pos_path = os.path.join(OUTPUT_DIR, "position_feature_matrix.csv")
    pos_matrix.to_csv(pos_path, index=False)
    print(f"Position-level matrix saved → {pos_path} ({len(pos_matrix)} positions)")
    print(f"  Majority-pathogenic positions : {pos_matrix.majority_pathogenic.sum()}")
    print(f"  Majority-benign positions     : {(pos_matrix.majority_pathogenic==0).sum()}")

    # ── Classification ────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"Classification  ({N_FOLDS}-fold, GroupKFold by position)")
    print(f"{'='*62}")

    X_all  = vm[FEATURES].values
    y_var  = vm["label_bin"].values
    groups = vm["position"].values     # ensures same position never splits across folds

    X_hs   = vm[["is_hotspot"]].values
    X_full_nodna = vm[[f for f in FEATURES if f != "min_dist_dna"]].values

    # --- Model A: Full patch + DNA distance (12 features)
    y_prob_full, y_pred_full, auc_full, f1_full = run_classification(
        X_all, y_var, groups, "Full patch + DNA", N_FOLDS
    )

    # --- Model B: Patch without DNA distance (11 features)
    y_prob_nodna, y_pred_nodna, auc_nodna, f1_nodna = run_classification(
        X_full_nodna, y_var, groups, "Patch without DNA", N_FOLDS
    )

    # --- Model C: Hotspot membership only (1 feature, baseline)
    y_prob_hs, y_pred_hs, auc_hs, f1_hs = run_classification(
        X_hs, y_var, groups, "Hotspot only", N_FOLDS
    )

    # --- Model D: Random baseline
    dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    from sklearn.model_selection import cross_val_predict as cvp
    cv_gkf = StratifiedGroupKFold(n_splits=N_FOLDS)
    y_prob_rand = cvp(dummy, X_all, y_var, cv=cv_gkf,
                      method="predict_proba", groups=groups)[:, 1]
    auc_rand = roc_auc_score(y_var, y_prob_rand)

    # --- Position-level classification (no pseudo-replication at all)
    X_pos = pos_matrix[FEATURES].values
    y_pos = pos_matrix["majority_pathogenic"].values
    # Simple 5-fold stratified (positions are independent)
    from sklearn.model_selection import StratifiedKFold
    cv_pos = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pipe_pos = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                      class_weight="balanced")),
    ])
    y_prob_pos = cross_val_predict(pipe_pos, X_pos, y_pos, cv=cv_pos,
                                   method="predict_proba")[:, 1]
    y_pred_pos = (y_prob_pos >= 0.5).astype(int)
    auc_pos = roc_auc_score(y_pos, y_prob_pos)
    f1_pos  = f1_score(y_pos, y_pred_pos)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'Model':<40} {'AUC':>6}  {'F1':>6}  {'N':>5}")
    print("-" * 60)
    print(f"{'Full patch + DNA dist (12 feat)':40} {auc_full:6.4f}  {f1_full:6.4f}  {len(y_var):>5}")
    print(f"{'Patch without DNA (11 feat)':40} {auc_nodna:6.4f}  {f1_nodna:6.4f}  {len(y_var):>5}")
    print(f"{'Hotspot membership only (1 feat)':40} {auc_hs:6.4f}  {f1_hs:6.4f}  {len(y_var):>5}")
    print(f"{'Random baseline':40} {auc_rand:6.4f}    N/A  {len(y_var):>5}")
    print(f"{'Position-level (188 pos, no pseudo-rep)':40} {auc_pos:6.4f}  {f1_pos:6.4f}  {len(y_pos):>5}")
    print()
    print(f"Adding DNA distance:     delta-AUC = {auc_full - auc_nodna:+.4f}")
    print(f"Full patch vs hotspot:   delta-AUC = {auc_full - auc_hs:+.4f}")
    print(f"Full patch vs random:    delta-AUC = {auc_full - auc_rand:+.4f}")

    print(f"\nDetailed report — Full patch + DNA model (variant level):")
    print(classification_report(y_var, y_pred_full,
                                 target_names=["Benign","Pathogenic"]))

    cm = confusion_matrix(y_var, y_pred_full)
    print("Confusion matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    print(f"\nDetailed report — Position-level model:")
    print(classification_report(y_pos, y_pred_pos,
                                 target_names=["Majority benign","Majority pathogenic"]))

    # ── Feature importance ────────────────────────────────────────────────────
    scaler_fit = StandardScaler().fit(X_all)
    lr_fit     = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                     class_weight="balanced")
    lr_fit.fit(scaler_fit.transform(X_all), y_var)
    coef_df = pd.DataFrame({
        "feature":     FEATURES,
        "coefficient": lr_fit.coef_[0],
    }).sort_values("coefficient", key=abs, ascending=False)

    print("\nFeature importances (standardised logistic regression coefficients):")
    print(coef_df.to_string(index=False))

    # ── Save report ───────────────────────────────────────────────────────────
    lines = [
        "Patch Classification Results — TP53 Proof of Concept",
        "="*56,
        f"Variant-level analysis : {len(vm)} variants, {vm.position.nunique()} unique positions",
        f"Position-level analysis: {len(pos_matrix)} positions",
        f"Cross-validation       : {N_FOLDS}-fold GroupKFold (grouped by position)",
        f"Features               : {FEATURES}",
        "",
        f"{'Model':<40} {'AUC':>6}  {'F1':>6}",
        "-"*56,
        f"{'Full patch + DNA dist (12 feat)':40} {auc_full:6.4f}  {f1_full:6.4f}",
        f"{'Patch without DNA (11 feat)':40} {auc_nodna:6.4f}  {f1_nodna:6.4f}",
        f"{'Hotspot membership only (1 feat)':40} {auc_hs:6.4f}  {f1_hs:6.4f}",
        f"{'Random baseline':40} {auc_rand:6.4f}",
        f"{'Position-level (no pseudo-rep)':40} {auc_pos:6.4f}  {f1_pos:6.4f}",
        "",
        "delta-AUC (DNA feature added)  : " + f"{auc_full - auc_nodna:+.4f}",
        "delta-AUC (full vs hotspot-only): " + f"{auc_full - auc_hs:+.4f}",
        "",
        "Detailed report — Full patch + DNA (variant level):",
        classification_report(y_var, y_pred_full,
                               target_names=["Benign","Pathogenic"]),
        "",
        "Feature coefficients:",
        coef_df.to_string(index=False),
    ]
    rpt_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(rpt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nReport saved → {rpt_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC curves
    ax = axes[0]
    for y_p, lbl, col, ls in [
        (y_prob_full,  f"Full patch + DNA  (AUC={auc_full:.3f})",  "#d62728",  "-"),
        (y_prob_nodna, f"Patch w/o DNA     (AUC={auc_nodna:.3f})", "#ff7f0e",  "--"),
        (y_prob_hs,    f"Hotspot only      (AUC={auc_hs:.3f})",    "#1f77b4",  "-."),
        (y_prob_rand,  f"Random            (AUC={auc_rand:.3f})",  "gray",     ":"),
    ]:
        fpr, tpr, _ = roc_curve(y_var, y_p)
        ax.plot(fpr, tpr, label=lbl, color=col, linestyle=ls, linewidth=2)
    ax.plot([0,1],[0,1],"k:",linewidth=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC curves\n5-fold GroupKFold CV (grouped by position)", fontsize=10)
    ax.legend(fontsize=8)

    # Feature importance
    ax2 = axes[1]
    colors = ["#d62728" if c > 0 else "#1f77b4" for c in coef_df["coefficient"]]
    ax2.barh(coef_df["feature"], coef_df["coefficient"], color=colors, alpha=0.85)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Logistic regression coefficient\n(positive = pathogenic signal)", fontsize=10)
    ax2.set_title("Feature importance\n(standardised coefficients, full model)", fontsize=10)
    ax2.invert_yaxis()

    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"ROC + importance plot saved → {roc_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
