#!/usr/bin/env python3
"""
compare_tools.py
================
Phase 2 of the improvement pipeline:
  1. Download AlphaMissense scores for TP53 from EBI
  2. Load our best model predictions (from improved_feature_matrix.csv)
  3. Bootstrap 95% CI for AUC-ROC (1000 iterations)
  4. DeLong test: patch model vs AlphaMissense
  5. Head-to-head ROC curves + summary table

RUN
  python3 compare_tools.py

OUTPUT  (derived/)
  comparison_roc.png
  comparison_report.txt
  bootstrap_ci_results.csv
"""

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET SUBSET — ~437 VARIANTS (intersection with AlphaMissense scores)
# ═══════════════════════════════════════════════════════════════════════════════
# The improved feature matrix contains 1,374 variants. This script restricts
# the comparison to variants for which an AlphaMissense pathogenicity score
# is available (matched by exact position + wt_aa + mut_aa triple).
# This intersection yields ~437 variants.
#
# WHY A SUBSET:
#   AlphaMissense scores are retrieved from alphamisssense_tp53.csv, which
#   covers all theoretically possible TP53 missense variants. The subset is
#   smaller than 1,374 because:
#     1. The "Protein change" column in the feature matrix is needed for the
#        wt/mut amino-acid lookup; entries where this parsing fails are excluded.
#     2. Variants at positions outside AlphaMissense coverage are excluded.
#
# CRITICAL: All AUC values in this comparison — for both AlphaMissense AND
# the patch models — are computed on this SAME ~437-variant subset. This is
# the only valid basis for a head-to-head comparison. The patch model AUC
# on this subset (~0.89) is higher than on the full 1,374-variant set (~0.82)
# because the subset has a different variant composition; do not conflate them.
# ═══════════════════════════════════════════════════════════════════════════════

import os, re, sys, warnings, urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT     = os.path.join(HERE, "derived")
FM_CSV  = os.path.join(OUT, "improved_feature_matrix.csv")
AM_URL  = "https://alphafold.ebi.ac.uk/files/AF-P04637-F1-aa-substitutions.csv"
AM_FILE = os.path.join(HERE, "alphamisssense_tp53.csv")

SEED      = 42
N_BOOT    = 1000
N_FOLDS   = 5

THREE_TO_ONE = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_auc(y_true, y_score, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs = np.array(aucs)
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


# ─────────────────────────────────────────────────────────────────────────────
# DeLong test (non-parametric AUC comparison)
# ─────────────────────────────────────────────────────────────────────────────
def delong_test(y_true, score_a, score_b):
    """Returns z-statistic and two-sided p-value for H0: AUC_A = AUC_B."""
    def auc_variance(y_true, scores):
        pos = scores[y_true == 1]
        neg = scores[y_true == 0]
        n1, n0 = len(pos), len(neg)
        V10 = np.mean([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])
        V01 = np.mean([np.mean(n < pos) + 0.5 * np.mean(n == pos) for n in neg])
        Q1 = np.mean([(np.mean(p > neg) + 0.5 * np.mean(p == neg))**2 for p in pos])
        Q2 = np.mean([(np.mean(n < pos) + 0.5 * np.mean(n == pos))**2 for n in neg])
        var = (V10*(1-V10) + (n1-1)*(Q1-V10**2) + (n0-1)*(Q2-V01**2)) / (n1*n0)
        return var
    auc_a = roc_auc_score(y_true, score_a)
    auc_b = roc_auc_score(y_true, score_b)
    var_a = auc_variance(y_true, score_a)
    var_b = auc_variance(y_true, score_b)
    diff  = auc_a - auc_b
    se    = np.sqrt(var_a + var_b)
    z     = diff / se if se > 0 else 0.0
    from scipy.stats import norm
    p     = 2 * (1 - norm.cdf(abs(z)))
    return z, p


# ─────────────────────────────────────────────────────────────────────────────
# Parse protein change
# ─────────────────────────────────────────────────────────────────────────────
def parse_pc(pc):
    if pd.isna(pc): return None, None, None
    pc = str(pc).strip().lstrip("p.")
    m = re.match(r'^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', pc)
    if m:
        wt3, pos, mut3 = m.groups()
        return THREE_TO_ONE.get(wt3.upper()), int(pos), THREE_TO_ONE.get(mut3.upper())
    m = re.match(r'^([A-CDEFGHIKLMNPQRSTVWY])(\d+)([A-CDEFGHIKLMNPQRSTVWY])$', pc)
    if m:
        wt1, pos, mut1 = m.groups()
        return wt1, int(pos), mut1
    return None, None, None


def make_pipeline(clf):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     clf),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 68)
    print("  TP53 Patch Model vs AlphaMissense — Comparison + Bootstrap CI")
    print("=" * 68)

    # ── 1. Load feature matrix from improved pipeline ─────────────────────────
    print("\n[1/5] Loading improved feature matrix...")
    if not os.path.exists(FM_CSV):
        print(f"  ERROR: {FM_CSV} not found. Run tp53_improved_classification.py first.")
        sys.exit(1)

    fm = pd.read_csv(FM_CSV)
    print(f"  Variants: {len(fm)}")

    POSITION_FEATS = [
        "patch_size","mean_hydrophobicity","frac_positive","frac_negative",
        "frac_neutral","mean_relative_sasa","mean_residue_size","frac_buried",
        "frac_hydrophobic","mean_ddg","std_ddg","frac_destabilizing","max_ddg",
        "is_hotspot","min_dist_dna","bfactor_mean","secondary_structure",
        "zinc_distance","plddt","specific_ddg","ddg_target_aa","blosum62_score",
        "delta_hydrophobicity","delta_size","charge_change",
    ]
    feature_cols = [c for c in POSITION_FEATS if c in fm.columns]
    X      = fm[feature_cols].values.astype(float)
    y      = fm["label_bin"].values
    groups = fm["position"].values

    # ── 2. Download + merge AlphaMissense ─────────────────────────────────────
    print("\n[2/5] Loading AlphaMissense scores...")
    if not os.path.exists(AM_FILE):
        print(f"  Downloading from EBI...")
        urllib.request.urlretrieve(AM_URL, AM_FILE)
    am = pd.read_csv(AM_FILE)
    print(f"  AlphaMissense entries: {len(am)}")

    # Parse protein_variant (e.g. "R175H") → position + mut_aa
    am["wt_aa"]  = am["protein_variant"].str[0]
    am["mut_aa"] = am["protein_variant"].str[-1]
    am["position"] = am["protein_variant"].str[1:-1].astype(int)

    # Parse ClinVar protein changes to get wt/mut for merge
    pc_parsed = fm["Protein change"].apply(parse_pc) if "Protein change" in fm.columns else None

    if pc_parsed is not None:
        fm["wt_aa_cv"]  = pc_parsed.apply(lambda x: x[0])
        fm["mut_aa_cv"] = pc_parsed.apply(lambda x: x[2])
    else:
        # Try to recover from existing columns
        fm["wt_aa_cv"]  = None
        fm["mut_aa_cv"] = None

    # Merge on position + wt_aa + mut_aa
    fm_am = fm.merge(
        am[["position","wt_aa","mut_aa","am_pathogenicity","am_class"]],
        left_on=["position","wt_aa_cv","mut_aa_cv"],
        right_on=["position","wt_aa","mut_aa"],
        how="left",
    )
    n_am = fm_am["am_pathogenicity"].notna().sum()
    print(f"  Variants with AM score: {n_am}/{len(fm_am)}")

    # Restrict comparison to variants that have AlphaMissense scores
    shared = fm_am[fm_am["am_pathogenicity"].notna()].copy()
    print(f"  Shared variants for comparison: {len(shared)}")
    if len(shared) < 50:
        print("  WARNING: too few shared variants — comparison may be unreliable.")

    X_shared      = shared[feature_cols].values.astype(float)
    y_shared      = shared["label_bin"].values
    groups_shared = shared["position"].values
    am_scores     = shared["am_pathogenicity"].values

    # ── 3. Cross-validated predictions for our ensemble on shared subset ───────
    print("\n[3/5] Computing cross-validated predictions (StratifiedGroupKFold)...")
    cv = StratifiedGroupKFold(n_splits=N_FOLDS)

    rf = make_pipeline(RandomForestClassifier(
        n_estimators=400, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, max_features=0.5, class_weight="balanced",
        random_state=SEED))
    gb = make_pipeline(GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=5,
        subsample=0.8, min_samples_leaf=5, random_state=SEED))
    svm = make_pipeline(SVC(probability=True, kernel="rbf", C=10,
        gamma=0.01, class_weight="balanced", random_state=SEED))

    ensemble = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf", VotingClassifier(
            estimators=[
                ("rf",  rf.named_steps["clf"]),
                ("gb",  gb.named_steps["clf"]),
                ("svm", svm.named_steps["clf"]),
            ], voting="soft")),
    ])
    lr = make_pipeline(LogisticRegression(
        max_iter=3000, class_weight="balanced", C=1.0, random_state=SEED))

    models = {
        "Logistic Regression": lr,
        "Random Forest":       rf,
        "Gradient Boosting":   gb,
        "Ensemble (RF+GB+SVM)": ensemble,
    }
    pred_probs = {}
    for name, model in models.items():
        probs = np.zeros(len(shared))
        valid = np.zeros(len(shared), dtype=bool)
        try:
            for train_idx, test_idx in cv.split(X_shared, y_shared, groups_shared):
                model.fit(X_shared[train_idx], y_shared[train_idx])
                probs[test_idx] = model.predict_proba(X_shared[test_idx])[:, 1]
                valid[test_idx] = True
            pred_probs[name] = probs
            auc = roc_auc_score(y_shared[valid], probs[valid])
            print(f"  {name}: AUC = {auc:.4f}")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    # ── 4. Bootstrap CI + DeLong test ─────────────────────────────────────────
    print(f"\n[4/5] Bootstrap 95% CI ({N_BOOT} iterations) + DeLong test...")
    results = {}

    # AlphaMissense
    am_auc = roc_auc_score(y_shared, am_scores)
    am_mean, am_lo, am_hi = bootstrap_auc(y_shared, am_scores, n=N_BOOT, seed=SEED)
    am_apr = average_precision_score(y_shared, am_scores)
    results["AlphaMissense"] = {
        "AUC": am_auc, "CI_lo": am_lo, "CI_hi": am_hi,
        "AUC-PR": am_apr, "delong_z": None, "delong_p": None,
    }
    print(f"  AlphaMissense: AUC = {am_auc:.4f}  95% CI [{am_lo:.4f}, {am_hi:.4f}]")

    best_name, best_auc = None, 0
    for name, probs in pred_probs.items():
        auc   = roc_auc_score(y_shared, probs)
        mean, lo, hi = bootstrap_auc(y_shared, probs, n=N_BOOT, seed=SEED)
        apr   = average_precision_score(y_shared, probs)
        z, p  = delong_test(y_shared, probs, am_scores)
        results[name] = {
            "AUC": auc, "CI_lo": lo, "CI_hi": hi,
            "AUC-PR": apr, "delong_z": z, "delong_p": p,
        }
        print(f"  {name}: AUC = {auc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
              f"vs AM: z={z:+.2f}  p={p:.4f}")
        if auc > best_auc:
            best_auc, best_name = auc, name

    # ── 5. Plots + report ──────────────────────────────────────────────────────
    print("\n[5/5] Generating comparison plot and report...")

    palette = {
        "AlphaMissense":       "#e41a1c",
        "Ensemble (RF+GB+SVM)":"#377eb8",
        "Random Forest":       "#4daf4a",
        "Gradient Boosting":   "#ff7f00",
        "Logistic Regression": "#984ea3",
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0,1],[0,1],"k--",lw=0.8,label="Random (AUC = 0.50)")

    # AlphaMissense
    fpr_am, tpr_am, _ = roc_curve(y_shared, am_scores)
    ax.plot(fpr_am, tpr_am, color=palette["AlphaMissense"], lw=2.5,
            label=f"AlphaMissense  AUC = {am_auc:.3f} [{am_lo:.3f}–{am_hi:.3f}]")

    # Patch models
    for name, probs in pred_probs.items():
        r = results[name]
        fpr, tpr, _ = roc_curve(y_shared, probs)
        p_str = f"p={r['delong_p']:.3f}" if r["delong_p"] is not None else ""
        ax.plot(fpr, tpr, color=palette.get(name,"grey"), lw=2,
                linestyle="--" if name != best_name else "-",
                label=f"{name}  AUC = {r['AUC']:.3f} [{r['CI_lo']:.3f}–{r['CI_hi']:.3f}]  {p_str}")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("TP53 Patch Model vs AlphaMissense — ROC Comparison\n"
                 f"(n={len(shared)} variants with AM score)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(OUT, "comparison_roc.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"  ROC plot → {roc_path}")

    # Bootstrap CI bar chart
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    names_plot = ["AlphaMissense"] + list(pred_probs.keys())
    aucs_plot  = [results[n]["AUC"]   for n in names_plot]
    lo_plot    = [results[n]["AUC"] - results[n]["CI_lo"] for n in names_plot]
    hi_plot    = [results[n]["CI_hi"] - results[n]["AUC"] for n in names_plot]
    colors     = [palette.get(n,"grey") for n in names_plot]
    bars = ax2.barh(names_plot[::-1], aucs_plot[::-1],
                    xerr=[lo_plot[::-1], hi_plot[::-1]],
                    color=colors[::-1], edgecolor="black", linewidth=0.5,
                    capsize=4, height=0.5)
    ax2.axvline(0.5, color="grey", lw=1, linestyle="--")
    ax2.set_xlabel("AUC-ROC (95% Bootstrap CI)", fontsize=11)
    ax2.set_title("AUC comparison with 95% CI", fontsize=11)
    ax2.set_xlim(0.45, 1.0)
    for i, (auc, name) in enumerate(zip(aucs_plot[::-1], names_plot[::-1])):
        ax2.text(auc + hi_plot[::-1][i] + 0.005, i, f"{auc:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    ci_path = os.path.join(OUT, "comparison_ci_bars.png")
    plt.savefig(ci_path, dpi=150)
    plt.close()
    print(f"  CI bar chart → {ci_path}")

    # CSV summary
    ci_df = pd.DataFrame([
        {"Model": n, "AUC": f"{r['AUC']:.4f}",
         "CI_95_lo": f"{r['CI_lo']:.4f}", "CI_95_hi": f"{r['CI_hi']:.4f}",
         "AUC_PR": f"{r['AUC-PR']:.4f}",
         "DeLong_z_vs_AM": f"{r['delong_z']:+.3f}" if r["delong_z"] else "—",
         "DeLong_p_vs_AM": f"{r['delong_p']:.4f}" if r["delong_p"] else "—"}
        for n, r in results.items()
    ])
    ci_csv = os.path.join(OUT, "bootstrap_ci_results.csv")
    ci_df.to_csv(ci_csv, index=False)
    print(f"  Bootstrap CI table → {ci_csv}")

    # Text report
    lines = [
        "TP53 Structural Patch Model — Comparison with AlphaMissense",
        "=" * 60,
        f"Comparison dataset: {len(shared)} variants with AlphaMissense score",
        f"  Pathogenic: {y_shared.sum()} ({y_shared.mean():.1%})",
        f"  Benign    : {(1-y_shared).sum()} ({(1-y_shared).mean():.1%})",
        f"Bootstrap: {N_BOOT} iterations, seed={SEED}",
        f"Cross-validation: {N_FOLDS}-fold StratifiedGroupKFold (by position)",
        "",
        f"{'Model':<30} {'AUC':>6} {'95% CI':>18} {'AUC-PR':>7} {'DeLong z':>10} {'p-value':>9}",
        "-" * 85,
    ]
    for n, r in results.items():
        ci_str  = f"[{r['CI_lo']:.4f}–{r['CI_hi']:.4f}]"
        z_str   = f"{r['delong_z']:+.3f}" if r["delong_z"] is not None else "  —   "
        p_str   = f"{r['delong_p']:.4f}"  if r["delong_p"] is not None else "  —   "
        lines.append(f"{n:<30} {r['AUC']:6.4f} {ci_str:>18} {r['AUC-PR']:7.4f} {z_str:>10} {p_str:>9}")
    lines += [
        "",
        "DeLong test: H0 = patch model AUC equals AlphaMissense AUC",
        "  p < 0.05 → statistically significant difference in AUC",
        "",
        f"Best patch model: {best_name}  (AUC = {best_auc:.4f})",
        "",
        "Note: AlphaMissense is a large-scale deep learning model trained on",
        "evolutionary and structural data. Our model uses only local 3D patch",
        "descriptors without sequence conservation — comparison is intentional,",
        "as the goal is to assess the independent contribution of structural",
        "microenvironment features.",
    ]
    rpt_path = os.path.join(OUT, "comparison_report.txt")
    with open(rpt_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"  Report → {rpt_path}")

    print("\n" + "=" * 68)
    print("  Done.")
    print("=" * 68)


if __name__ == "__main__":
    main()
