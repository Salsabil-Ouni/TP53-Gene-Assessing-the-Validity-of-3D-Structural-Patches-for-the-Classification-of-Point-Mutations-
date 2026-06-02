"""
feature_ablation.py
===================
Ablation study — isolating the contribution of 3D structural patch features
vs. sequence-level substitution features vs. thermodynamic stability features.

This is the experiment that directly answers the central thesis question:
  "Do 3D structural patch descriptors carry discriminative signal for
   missense variant pathogenicity BEYOND what can be derived from
   amino-acid sequence properties alone?"

Six feature subsets are evaluated with identical experimental conditions
(GradientBoosting, 5-fold StratifiedGroupKFold grouped by residue position)
so that AUC differences between groups are attributable only to the features.

Feature group definitions
--------------------------
SEQ_ONLY (4 features)
    Properties computable from the substitution identity alone — no 3D
    structure required.
    blosum62_score, delta_hydrophobicity, delta_size, charge_change

PATCH_ONLY (10 features)
    Local 3D microenvironment descriptors: the neighbourhood within 6 Å of
    the Cα of the central residue (BioPython NeighborSearch on tp53_Repair.pdb).
    These are the features the thesis hypothesis specifically refers to.
    patch_size, mean_hydrophobicity, frac_positive, frac_negative,
    frac_neutral, mean_relative_sasa, mean_residue_size, frac_buried,
    frac_hydrophobic, min_dist_dna

STABILITY_ONLY (7 features)
    Thermodynamic profile from FoldX PositionScan: position-level and
    mutation-specific free-energy estimates.
    mean_ddg, std_ddg, frac_destabilizing, max_ddg, is_hotspot,
    specific_ddg, ddg_target_aa

PATCH+STABILITY (17 features)
    Full structural signal: local 3D environment + thermodynamic context.
    No sequence substitution features.

SEQ+PATCH (14 features)
    Local structural environment + sequence features; no stability features.
    Tests whether sequence adds to pure structural patch signal.

FULL_25FEAT (25 features)
    All features — replicates §3.6 Gradient Boosting result as reference.

Key comparisons
---------------
seq_only vs patch_only          — does structure beat sequence?
patch_only vs patch+stability   — does adding FoldX ΔΔG help?
patch+stability vs full         — marginal value of sequence when structure present?
seq_only vs full                — total gain from structural features

INPUT:
    derived/improved_feature_matrix.csv   (produced by tp53_improved_classification.py)

OUTPUT:
    derived/ablation_results.csv
    derived/ablation_report.txt
    derived/ablation_roc_curves.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

SEQ_FEATURES = [
    "blosum62_score",       # BLOSUM62 substitution score (evolutionary proxy)
    "delta_hydrophobicity", # KD(mutant) − KD(wild-type)
    "delta_size",           # heavy-atom count change upon substitution
    "charge_change",        # binary: does the substitution change charge?
]

PATCH_FEATURES = [
    "patch_size",           # number of neighbours within 6 Å of Cα
    "mean_hydrophobicity",  # mean Kyte-Doolittle hydrophobicity of neighbourhood
    "frac_positive",        # fraction positively charged neighbours
    "frac_negative",        # fraction negatively charged neighbours
    "frac_neutral",         # fraction neutral neighbours
    "mean_relative_sasa",   # mean relative solvent-accessible surface area
    "mean_residue_size",    # mean heavy-atom count of neighbour residues
    "frac_buried",          # fraction of neighbours with rSASA < 0.20
    "frac_hydrophobic",     # fraction of neighbours with KD > 0
    "min_dist_dna",         # minimum distance to DNA heavy atom (PDB 3KZ8)
]

STABILITY_FEATURES = [
    "mean_ddg",             # position mean ΔΔG across all 19 substitutions
    "std_ddg",              # standard deviation of position ΔΔG
    "frac_destabilizing",   # fraction of substitutions with ΔΔG > 0
    "max_ddg",              # maximum ΔΔG at position
    "is_hotspot",           # binary: mean ΔΔG ≥ 1.5 kcal/mol
    "specific_ddg",         # FoldX ΔΔG for this specific substitution
    "ddg_target_aa",        # target-AA-encoded FoldX ΔΔG
]

STRUCTURAL_CONTEXT = [
    "zinc_distance",        # distance to Zn²⁺ ion (Å)
    "bfactor_mean",         # mean B-factor of patch residues (flexibility)
    "secondary_structure",  # 0 = coil, 1 = helix, 2 = sheet
    "plddt",                # AlphaFold2 pLDDT (structural confidence)
]

ALL_25 = (SEQ_FEATURES + PATCH_FEATURES + STABILITY_FEATURES +
          STRUCTURAL_CONTEXT)

FEATURE_GROUPS = {
    "seq_only":         SEQ_FEATURES,
    "patch_only":       PATCH_FEATURES,
    "stability_only":   STABILITY_FEATURES,
    "patch+stability":  PATCH_FEATURES + STABILITY_FEATURES,
    "seq+patch":        SEQ_FEATURES + PATCH_FEATURES,
    "full_25feat":      ALL_25,
}

OUTPUT_DIR = "derived"
FEATURE_MATRIX = os.path.join(OUTPUT_DIR, "improved_feature_matrix.csv")

GB_PARAMS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, random_state=42,
)


def run_cv(X, y, groups):
    """5-fold StratifiedGroupKFold with GradientBoosting + median imputation."""
    cv = StratifiedGroupKFold(n_splits=5)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     GradientBoostingClassifier(**GB_PARAMS)),
    ])
    probas = cross_val_predict(
        pipe, X, y, cv=cv, groups=groups, method="predict_proba"
    )[:, 1]
    preds   = (probas >= 0.5).astype(int)
    auc_roc = roc_auc_score(y, probas)
    auc_pr  = average_precision_score(y, probas)
    f1      = f1_score(y, preds, zero_division=0)
    fpr, tpr, _ = roc_curve(y, probas)
    return auc_roc, auc_pr, f1, fpr, tpr, probas


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(FEATURE_MATRIX):
        print(f"[ERROR] {FEATURE_MATRIX} not found.")
        print("Run tp53_improved_classification.py first.")
        return

    fm = pd.read_csv(FEATURE_MATRIX)
    fm.columns = fm.columns.str.strip()

    y      = fm["label_bin"].values
    groups = fm["position"].values

    print("=" * 70)
    print("Feature Ablation Study — Structural vs Sequence Contributions")
    print("=" * 70)
    print(f"\nVariants: {len(fm)}  |  "
          f"Pathogenic: {y.sum()}  |  Benign: {(y == 0).sum()}")
    print(f"Cross-validation: 5-fold StratifiedGroupKFold (by position)\n")

    results  = []
    roc_data = {}

    for name, feats in FEATURE_GROUPS.items():
        available = [f for f in feats if f in fm.columns]
        missing   = [f for f in feats if f not in fm.columns]
        if missing:
            print(f"  [{name}] columns not found: {missing}")

        X = fm[available].values
        auc_roc, auc_pr, f1, fpr, tpr, _ = run_cv(X, y, groups)

        results.append({
            "group":      name,
            "n_features": len(available),
            "auc_roc":    round(auc_roc, 4),
            "auc_pr":     round(auc_pr,  4),
            "f1":         round(f1,      4),
        })
        roc_data[name] = (fpr, tpr, auc_roc)

        print(f"  {name:<22}  n={len(available):2d}  "
              f"AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  F1={f1:.4f}")

    # Compute deltas relative to seq_only
    seq_auc = next(r["auc_roc"] for r in results if r["group"] == "seq_only")
    for r in results:
        r["delta_auc_vs_seq"] = round(r["auc_roc"] - seq_auc, 4)

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    res_df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\nAblation results saved -> {csv_path}")

    # -----------------------------------------------------------------------
    # Save text report
    # -----------------------------------------------------------------------
    lines = [
        "TP53 Structural Patch — Feature Ablation Report",
        "=" * 70,
        "",
        "Classifier : GradientBoosting (n_estimators=200, max_depth=3, lr=0.05)",
        "CV         : 5-fold StratifiedGroupKFold (grouped by residue position)",
        "Variants   : 1,374  (874 pathogenic, 500 benign)",
        "Imputation : median (for NaN at positions outside FoldX/patch domain)",
        "",
        "Feature groups:",
        f"  seq_only       ({len(SEQ_FEATURES):2d} feat): {SEQ_FEATURES}",
        f"  patch_only     ({len(PATCH_FEATURES):2d} feat): {PATCH_FEATURES}",
        f"  stability_only ({len(STABILITY_FEATURES):2d} feat): {STABILITY_FEATURES}",
        f"  patch+stability ({len(PATCH_FEATURES+STABILITY_FEATURES):2d} feat)",
        f"  seq+patch       ({len(SEQ_FEATURES+PATCH_FEATURES):2d} feat)",
        f"  full_25feat     ({len(ALL_25):2d} feat)",
        "",
        f"{'Group':<24} {'Feat':>4} {'AUC-ROC':>8} {'AUC-PR':>8} "
        f"{'F1':>6} {'ΔAUC vs seq':>12}",
        "-" * 70,
    ]
    for r in results:
        sign = "+" if r["delta_auc_vs_seq"] >= 0 else ""
        lines.append(
            f"  {r['group']:<22} {r['n_features']:>4} {r['auc_roc']:>8.4f} "
            f"{r['auc_pr']:>8.4f} {r['f1']:>6.4f}  "
            f"{sign}{r['delta_auc_vs_seq']:>+.4f}"
        )
    lines += [
        "",
        "Key comparisons:",
        f"  seq_only     → patch_only     : "
        f"ΔAUC = {roc_data['patch_only'][2] - roc_data['seq_only'][2]:+.4f}  "
        "(3D structure vs sequence alone)",
        f"  patch_only   → patch+stability: "
        f"ΔAUC = {roc_data['patch+stability'][2] - roc_data['patch_only'][2]:+.4f}  "
        "(adding thermodynamic features to patch)",
        f"  seq+patch    → full_25feat    : "
        f"ΔAUC = {roc_data['full_25feat'][2] - roc_data['seq+patch'][2]:+.4f}  "
        "(marginal value of stability when sequence+patch present)",
        f"  seq_only     → full_25feat    : "
        f"ΔAUC = {roc_data['full_25feat'][2] - roc_data['seq_only'][2]:+.4f}  "
        "(total structural contribution over sequence baseline)",
    ]

    report_path = os.path.join(OUTPUT_DIR, "ablation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Ablation report saved  -> {report_path}")

    # -----------------------------------------------------------------------
    # ROC curve plot
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))

    style = {
        "seq_only":        ("#9E9E9E", "--", 1.8),
        "patch_only":      ("#2196F3", "-",  2.5),
        "stability_only":  ("#FF9800", "--", 1.8),
        "patch+stability": ("#4CAF50", "-",  2.2),
        "seq+patch":       ("#9C27B0", "-",  2.0),
        "full_25feat":     ("#F44336", "-",  2.5),
    }
    for name, (fpr, tpr, auc_val) in roc_data.items():
        color, ls, lw = style[name]
        ax.plot(fpr, tpr, color=color, lw=lw, ls=ls,
                label=f"{name}  (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5, label="Random (0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "Feature group ablation — ROC curves\n"
        "GradientBoosting · 5-fold StratifiedGroupKFold · n = 1,374 variants",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "ablation_roc_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"ROC curves plot saved  -> {plot_path}")

    # -----------------------------------------------------------------------
    # Summary printout
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    seq_auc_val   = roc_data["seq_only"][2]
    patch_auc_val = roc_data["patch_only"][2]
    full_auc_val  = roc_data["full_25feat"][2]
    ps_auc_val    = roc_data["patch+stability"][2]

    d_patch = patch_auc_val - seq_auc_val
    d_ps    = ps_auc_val - seq_auc_val
    d_full  = full_auc_val - seq_auc_val
    print(f"\n  Sequence-only baseline      AUC = {seq_auc_val:.4f}")
    print(f"  Patch-only (3D structure)   AUC = {patch_auc_val:.4f}  "
          f"(dAUC = {d_patch:+.4f} over sequence)")
    print(f"  Patch + stability           AUC = {ps_auc_val:.4f}  "
          f"(dAUC = {d_ps:+.4f} over sequence)")
    print(f"  Full 25-feature model       AUC = {full_auc_val:.4f}  "
          f"(dAUC = {d_full:+.4f} over sequence)")

    if patch_auc_val > seq_auc_val:
        print("\n  CONCLUSION: 3D structural patch features outperform "
              "sequence-only features.")
        print("  The local physicochemical microenvironment carries "
              "discriminative signal")
        print("  that cannot be recovered from substitution identity alone.")
    else:
        print("\n  NOTE: sequence features match or outperform patch features.")
        print("  This should be discussed carefully in the thesis.")

    print("\nDone.")


if __name__ == "__main__":
    main()
