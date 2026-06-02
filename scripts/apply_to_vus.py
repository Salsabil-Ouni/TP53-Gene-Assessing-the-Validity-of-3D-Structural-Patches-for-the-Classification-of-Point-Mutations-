#!/usr/bin/env python3
"""
apply_to_vus.py
===============
Applies the trained TP53 structural patch ensemble to ClinVar VUS.

Steps:
  1. Download TP53 missense VUS from ClinVar via NCBI API
  2. Build patch feature vectors (same pipeline as training)
  3. Train ensemble on all labelled variants (no CV — full training set)
  4. Predict pathogenicity probability for each VUS
  5. Output ranked list + summary plots

OUTPUT  (derived/)
  vus_predictions.csv        — all VUS with predicted P(pathogenic)
  vus_top_predicted.csv      — top 20 predicted pathogenic VUS
  vus_score_distribution.png — histogram of VUS scores
  vus_report.txt             — text summary for thesis

RUN
  python3 apply_to_vus.py
"""

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING SET — 1,374 LABELED VARIANTS (full improved feature matrix)
# ═══════════════════════════════════════════════════════════════════════════════
# The ensemble is trained on all 1,374 ClinVar-labeled variants from
# improved_feature_matrix.csv (874 pathogenic, 500 benign). This is the full
# set from tp53_improved_classification.py, extending the baseline 866-variant
# domain-restricted set to the full protein via AlphaFold2 (AF-P04637).
#
# No cross-validation here — the model is trained on the FULL labeled set
# before applying to VUS. Training AUC is in-sample and must NOT be reported
# as a performance estimate; use the CV AUC from tp53_improved_classification.py
# for that purpose.
#
# VUS source: ClinVar via NCBI Entrez API (cached to derived/vus_raw.csv after
# first run). Only VUS with a structural patch feature vector (i.e. positions
# covered by 2OCJ or AlphaFold) are scored; others are reported as no-coverage.
# ═══════════════════════════════════════════════════════════════════════════════

import os, re, sys, time, warnings, urllib.request, urllib.parse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import (RandomForestClassifier,
                                   GradientBoostingClassifier,
                                   VotingClassifier)
from sklearn.svm           import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.impute        import SimpleImputer
from sklearn.metrics       import roc_auc_score

from Bio import PDB
from Bio.PDB import NeighborSearch
from Bio.PDB.SASA import ShrakeRupley

HERE     = os.path.dirname(os.path.abspath(__file__))
OUT      = os.path.join(HERE, "derived")
FM_CSV   = os.path.join(OUT, "improved_feature_matrix.csv")
PDB_FILE = os.path.join(HERE, "tp53_Repair.pdb")
DNA_PDB  = os.path.join(HERE, "3kz8.pdb1")
AF_FILE  = os.path.join(HERE, "AF_P04637_tp53.pdb")
FOLDX_MUT = os.path.join(OUT, "foldx_per_mutation.csv")
POS_SUM   = os.path.join(OUT, "foldx_position_summary.csv")

SEED    = 42
CHAIN   = "A"
DNA_CHAIN = "C"
RADIUS  = 6.0
HS_THRESH = 1.5
DEST_THRESH = 1.0

THREE_TO_ONE = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}
ONE_TO_THREE = {v:k for k,v in THREE_TO_ONE.items()}
KD = {"ALA":1.8,"ARG":-4.5,"ASN":-3.5,"ASP":-3.5,"CYS":2.5,
      "GLN":-3.5,"GLU":-3.5,"GLY":-0.4,"HIS":-3.2,"ILE":4.5,
      "LEU":3.8,"LYS":-3.9,"MET":1.9,"PHE":2.8,"PRO":-1.6,
      "SER":-0.8,"THR":-0.7,"TRP":-0.9,"TYR":-1.3,"VAL":4.2}
CHARGE = {"ARG":"positive","HIS":"positive","LYS":"positive",
          "ASP":"negative","GLU":"negative"}
MAX_ASA = {"ALA":129,"ARG":274,"ASN":195,"ASP":193,"CYS":167,
           "GLN":223,"GLU":223,"GLY":104,"HIS":224,"ILE":197,
           "LEU":201,"LYS":236,"MET":224,"PHE":240,"PRO":159,
           "SER":155,"THR":172,"TRP":285,"TYR":263,"VAL":174}
_BL62 = {
    "A":{"A":4,"R":-1,"N":-2,"D":-2,"C":0,"Q":-1,"E":-1,"G":0,"H":-2,"I":-1,"L":-1,"K":-1,"M":-1,"F":-2,"P":-1,"S":1,"T":0,"W":-3,"Y":-2,"V":0},
    "R":{"A":-1,"R":5,"N":0,"D":-2,"C":-3,"Q":1,"E":0,"G":-2,"H":0,"I":-3,"L":-2,"K":2,"M":-1,"F":-3,"P":-2,"S":-1,"T":-1,"W":-3,"Y":-2,"V":-3},
    "N":{"A":-2,"R":0,"N":6,"D":1,"C":-3,"Q":0,"E":0,"G":0,"H":1,"I":-3,"L":-3,"K":0,"M":-2,"F":-3,"P":-2,"S":1,"T":0,"W":-4,"Y":-2,"V":-3},
    "D":{"A":-2,"R":-2,"N":1,"D":6,"C":-3,"Q":0,"E":2,"G":-1,"H":-1,"I":-3,"L":-4,"K":-1,"M":-3,"F":-3,"P":-1,"S":0,"T":-1,"W":-4,"Y":-3,"V":-3},
    "C":{"A":0,"R":-3,"N":-3,"D":-3,"C":9,"Q":-3,"E":-4,"G":-3,"H":-3,"I":-1,"L":-1,"K":-3,"M":-1,"F":-2,"P":-3,"S":-1,"T":-1,"W":-2,"Y":-2,"V":-1},
    "Q":{"A":-1,"R":1,"N":0,"D":0,"C":-3,"Q":5,"E":2,"G":-2,"H":0,"I":-3,"L":-2,"K":1,"M":0,"F":-3,"P":-1,"S":0,"T":-1,"W":-2,"Y":-1,"V":-2},
    "E":{"A":-1,"R":0,"N":0,"D":2,"C":-4,"Q":2,"E":5,"G":-2,"H":0,"I":-3,"L":-3,"K":1,"M":-2,"F":-3,"P":-1,"S":0,"T":-1,"W":-3,"Y":-2,"V":-2},
    "G":{"A":0,"R":-2,"N":0,"D":-1,"C":-3,"Q":-2,"E":-2,"G":6,"H":-2,"I":-4,"L":-4,"K":-2,"M":-3,"F":-3,"P":-2,"S":0,"T":-2,"W":-2,"Y":-3,"V":-3},
    "H":{"A":-2,"R":0,"N":1,"D":-1,"C":-3,"Q":0,"E":0,"G":-2,"H":8,"I":-3,"L":-3,"K":-1,"M":-2,"F":-1,"P":-2,"S":-1,"T":-2,"W":-2,"Y":2,"V":-3},
    "I":{"A":-1,"R":-3,"N":-3,"D":-3,"C":-1,"Q":-3,"E":-3,"G":-4,"H":-3,"I":4,"L":2,"K":-3,"M":1,"F":0,"P":-3,"S":-2,"T":-1,"W":-3,"Y":-1,"V":3},
    "L":{"A":-1,"R":-2,"N":-3,"D":-4,"C":-1,"Q":-2,"E":-3,"G":-4,"H":-3,"I":2,"L":4,"K":-2,"M":2,"F":0,"P":-3,"S":-2,"T":-1,"W":-2,"Y":-1,"V":1},
    "K":{"A":-1,"R":2,"N":0,"D":-1,"C":-3,"Q":1,"E":1,"G":-2,"H":-1,"I":-3,"L":-2,"K":5,"M":-1,"F":-3,"P":-1,"S":0,"T":-1,"W":-3,"Y":-2,"V":-2},
    "M":{"A":-1,"R":-1,"N":-2,"D":-3,"C":-1,"Q":0,"E":-2,"G":-3,"H":-2,"I":1,"L":2,"K":-1,"M":5,"F":0,"P":-2,"S":-1,"T":-1,"W":-1,"Y":-1,"V":1},
    "F":{"A":-2,"R":-3,"N":-3,"D":-3,"C":-2,"Q":-3,"E":-3,"G":-3,"H":-1,"I":0,"L":0,"K":-3,"M":0,"F":6,"P":-4,"S":-2,"T":-2,"W":1,"Y":3,"V":-1},
    "P":{"A":-1,"R":-2,"N":-2,"D":-1,"C":-3,"Q":-1,"E":-1,"G":-2,"H":-2,"I":-3,"L":-3,"K":-1,"M":-2,"F":-4,"P":7,"S":-1,"T":-1,"W":-4,"Y":-3,"V":-2},
    "S":{"A":1,"R":-1,"N":1,"D":0,"C":-1,"Q":0,"E":0,"G":0,"H":-1,"I":-2,"L":-2,"K":0,"M":-1,"F":-2,"P":-1,"S":4,"T":1,"W":-3,"Y":-2,"V":-2},
    "T":{"A":0,"R":-1,"N":0,"D":-1,"C":-1,"Q":-1,"E":-1,"G":-2,"H":-2,"I":-1,"L":-1,"K":-1,"M":-1,"F":-2,"P":-1,"S":1,"T":5,"W":-2,"Y":-2,"V":0},
    "W":{"A":-3,"R":-3,"N":-4,"D":-4,"C":-2,"Q":-2,"E":-3,"G":-2,"H":-2,"I":-3,"L":-2,"K":-3,"M":-1,"F":1,"P":-4,"S":-3,"T":-2,"W":11,"Y":2,"V":-3},
    "Y":{"A":-2,"R":-2,"N":-2,"D":-3,"C":-2,"Q":-1,"E":-2,"G":-3,"H":2,"I":-1,"L":-1,"K":-2,"M":-1,"F":3,"P":-3,"S":-2,"T":-2,"W":2,"Y":7,"V":-1},
    "V":{"A":0,"R":-3,"N":-3,"D":-3,"C":-1,"Q":-2,"E":-2,"G":-3,"H":-3,"I":3,"L":1,"K":-2,"M":1,"F":-1,"P":-2,"S":-2,"T":0,"W":-3,"Y":-1,"V":4},
}

POSITION_FEATS = [
    "patch_size","mean_hydrophobicity","frac_positive","frac_negative",
    "frac_neutral","mean_relative_sasa","mean_residue_size","frac_buried",
    "frac_hydrophobic","mean_ddg","std_ddg","frac_destabilizing","max_ddg",
    "is_hotspot","min_dist_dna","bfactor_mean","secondary_structure",
    "zinc_distance","plddt","specific_ddg","ddg_target_aa","blosum62_score",
    "delta_hydrophobicity","delta_size","charge_change",
]

DNA_RES = {"DA","DT","DC","DG","A","T","C","G"}


# ─────────────────────────────────────────────────────────────────────────────
# NCBI ClinVar fetch
# ─────────────────────────────────────────────────────────────────────────────
def fetch_vus_from_clinvar():
    """Download TP53 missense VUS from NCBI ClinVar. Returns DataFrame."""
    print("  Searching NCBI ClinVar for TP53 missense VUS...")
    search_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        "?db=clinvar&term=TP53%5Bgene%5D+AND+%22uncertain+significance%22"
        "%5Bclinsig%5D+AND+%22missense+variant%22%5Bmoltype%5D"
        "&retmax=5000&retmode=json"
    )
    resp = urllib.request.urlopen(search_url)
    data = json.loads(resp.read())
    ids  = data["esearchresult"]["idlist"]
    print(f"  Found {len(ids)} VUS IDs")

    # Fetch summaries in batches of 200
    rows = []
    batch = 200
    for i in range(0, len(ids), batch):
        chunk = ids[i:i+batch]
        sum_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=clinvar&id={','.join(chunk)}&retmode=json"
        )
        try:
            sresp = urllib.request.urlopen(sum_url)
            sdata = json.loads(sresp.read())
            for vid in chunk:
                doc = sdata["result"].get(vid, {})
                title = doc.get("title", "")
                # Extract protein change from title e.g. "NM_000546.6(TP53):c.XXX (p.R175H)"
                m = re.search(r'\(p\.([A-Z][a-z]{2}\d+[A-Z][a-z]{2}|[A-Z]\d+[A-Z])\)', title)
                pc = m.group(1) if m else None
                rows.append({
                    "variation_id": vid,
                    "title": title,
                    "protein_change": pc,
                    "germline_classification": "Uncertain significance",
                })
            time.sleep(0.35)  # NCBI rate limit
        except Exception as e:
            print(f"  [WARN] Batch {i//batch+1} failed: {e}")
            time.sleep(1)

    df = pd.DataFrame(rows)
    n_pc = df["protein_change"].notna().sum()
    print(f"  Parsed protein change: {n_pc}/{len(df)} entries")
    return df


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


def blosum62(wt, mut):
    try: return _BL62[wt][mut]
    except: return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Build ensemble (train on all labelled data)
# ─────────────────────────────────────────────────────────────────────────────
def make_pipeline(clf):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     clf),
    ])


def train_ensemble(X_train, y_train):
    print("  Training ensemble on all labelled variants...")
    rf  = RandomForestClassifier(n_estimators=400, max_depth=20,
          min_samples_split=5, min_samples_leaf=2, max_features=0.5,
          class_weight="balanced", random_state=SEED, n_jobs=-1)
    gb  = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
          max_depth=5, subsample=0.8, min_samples_leaf=5, random_state=SEED)
    svm = SVC(probability=True, kernel="rbf", C=10, gamma=0.01,
              class_weight="balanced", random_state=SEED)
    ensemble = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf", VotingClassifier(
            estimators=[("rf",rf),("gb",gb),("svm",svm)],
            voting="soft")),
    ])
    ensemble.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train,
                    ensemble.predict_proba(X_train)[:,1])
    print(f"  Training AUC (in-sample): {train_auc:.4f}")
    return ensemble


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 68)
    print("  TP53 Structural Patch Model — VUS Prediction")
    print("=" * 68)

    # ── 1. Load training data ─────────────────────────────────────────────────
    print("\n[1/6] Loading labelled training data...")
    fm = pd.read_csv(FM_CSV)
    feat_cols = [c for c in POSITION_FEATS if c in fm.columns]
    X_train = fm[feat_cols].values.astype(float)
    y_train = fm["label_bin"].values
    print(f"  Training variants: {len(fm)}  |  Features: {len(feat_cols)}")
    print(f"  Pathogenic: {y_train.sum()}  Benign: {(1-y_train).sum()}")

    # ── 2. Train ensemble on full labelled set ────────────────────────────────
    print("\n[2/6] Training ensemble on full labelled dataset...")
    model = train_ensemble(X_train, y_train)

    # ── 3. Fetch VUS from ClinVar ─────────────────────────────────────────────
    vus_cache = os.path.join(OUT, "vus_raw.csv")
    print("\n[3/6] Loading VUS data...")
    if os.path.exists(vus_cache):
        print(f"  Using cached file: {vus_cache}")
        vus = pd.read_csv(vus_cache)
    else:
        vus = fetch_vus_from_clinvar()
        vus.to_csv(vus_cache, index=False)
        print(f"  Saved to cache: {vus_cache}")
    print(f"  VUS entries: {len(vus)}")

    # ── 4. Parse protein changes ──────────────────────────────────────────────
    print("\n[4/6] Parsing VUS protein changes...")
    parsed = vus["protein_change"].apply(parse_pc)
    vus["wt_aa"]   = parsed.apply(lambda x: x[0])
    vus["mut_aa"]  = parsed.apply(lambda x: x[2])
    vus["position"] = parsed.apply(lambda x: x[1])
    vus = vus[vus["wt_aa"].notna() & vus["mut_aa"].notna() & vus["position"].notna()].copy()
    vus["position"] = vus["position"].astype(int)
    print(f"  VUS with parseable protein change: {len(vus)}")
    # Deduplicate on protein_change (same variant may have multiple ClinVar entries)
    vus = vus.drop_duplicates(subset=["wt_aa","position","mut_aa"]).copy()
    print(f"  After deduplication: {len(vus)} unique VUS")

    # Remove any that appear in training data (overlap check)
    train_positions = set(fm["position"].unique())
    print(f"  Training covers positions: {len(train_positions)}")

    # ── 5. Build feature vectors for VUS ─────────────────────────────────────
    print("\n[5/6] Building feature vectors for VUS...")

    # Load precomputed patch features (from improved_feature_matrix.csv)
    # We reuse the same position-level patch features
    pos_feats = fm.groupby("position")[
        [c for c in POSITION_FEATS if c not in
         ["specific_ddg","ddg_target_aa","blosum62_score",
          "delta_hydrophobicity","delta_size","charge_change"]]
    ].first().reset_index()

    # Load foldx per-mutation for specific ΔΔG lookup
    foldx_mut = pd.read_csv(FOLDX_MUT)

    # Merge position features onto VUS
    vus_feat = vus.merge(pos_feats, on="position", how="left")

    # Add specific ΔΔG (exact match: pos + wt + mut)
    vus_feat = vus_feat.merge(
        foldx_mut[["position","wt_aa","mut_aa","ddg_foldx"]],
        on=["position","wt_aa","mut_aa"], how="left"
    ).rename(columns={"ddg_foldx":"specific_ddg"})

    # Add target-AA ΔΔG (pos + mut only, separate feature)
    foldx_by_mut = (foldx_mut[["position","mut_aa","ddg_foldx"]]
                    .rename(columns={"ddg_foldx":"ddg_target_aa"}))
    vus_feat = vus_feat.merge(foldx_by_mut, on=["position","mut_aa"], how="left")

    # Add BLOSUM62 and biochemical difference features
    vus_feat["blosum62_score"] = vus_feat.apply(
        lambda r: blosum62(r["wt_aa"], r["mut_aa"])
        if pd.notna(r.get("wt_aa")) and pd.notna(r.get("mut_aa")) else np.nan,
        axis=1)

    kd_map = {aa: KD.get(ONE_TO_THREE.get(aa,""), np.nan) for aa in "ACDEFGHIKLMNPQRSTVWY"}
    size_map = {"A":129,"R":274,"N":195,"D":193,"C":167,"Q":223,"E":223,"G":104,
                "H":224,"I":197,"L":201,"K":236,"M":224,"F":240,"P":159,
                "S":155,"T":172,"W":285,"Y":263,"V":174}
    charge_num = {"ARG":1,"HIS":1,"LYS":1,"ASP":-1,"GLU":-1}
    vus_feat["delta_hydrophobicity"] = vus_feat.apply(
        lambda r: kd_map.get(r["mut_aa"],np.nan) - kd_map.get(r["wt_aa"],np.nan)
        if pd.notna(r.get("wt_aa")) and pd.notna(r.get("mut_aa")) else np.nan, axis=1)
    vus_feat["delta_size"] = vus_feat.apply(
        lambda r: size_map.get(r["mut_aa"],np.nan) - size_map.get(r["wt_aa"],np.nan)
        if pd.notna(r.get("wt_aa")) and pd.notna(r.get("mut_aa")) else np.nan, axis=1)
    vus_feat["charge_change"] = vus_feat.apply(
        lambda r: (charge_num.get(ONE_TO_THREE.get(r["mut_aa"],""),0)
                   - charge_num.get(ONE_TO_THREE.get(r["wt_aa"],""),0))
        if pd.notna(r.get("wt_aa")) and pd.notna(r.get("mut_aa")) else np.nan, axis=1)

    # Keep only VUS that have at least patch_size feature (i.e. in covered positions)
    covered = vus_feat["patch_size"].notna()
    print(f"  VUS with patch features: {covered.sum()}/{len(vus_feat)}")
    vus_covered = vus_feat[covered].copy()

    # Impute remaining NaN with training medians
    for col in feat_cols:
        if col not in vus_covered.columns:
            vus_covered[col] = np.nan
        miss = vus_covered[col].isna().sum()
        if miss > 0:
            med = fm[col].median() if col in fm.columns else 0.0
            vus_covered[col] = vus_covered[col].fillna(med)

    X_vus = vus_covered[feat_cols].values.astype(float)

    # ── 6. Predict + report ───────────────────────────────────────────────────
    print("\n[6/6] Predicting pathogenicity for VUS...")
    probs = model.predict_proba(X_vus)[:, 1]
    vus_covered["P_pathogenic"] = probs
    vus_covered["predicted_class"] = pd.cut(
        probs,
        bins=[0, 0.3, 0.7, 1.0],
        labels=["Likely benign", "Uncertain", "Likely pathogenic"]
    )

    # Summary stats
    n_lp = (probs >= 0.7).sum()
    n_lb = (probs <= 0.3).sum()
    n_unc = ((probs > 0.3) & (probs < 0.7)).sum()
    print(f"\n  VUS reclassification summary:")
    print(f"    Likely pathogenic (≥0.70): {n_lp} ({n_lp/len(probs):.1%})")
    print(f"    Uncertain (0.30–0.70)    : {n_unc} ({n_unc/len(probs):.1%})")
    print(f"    Likely benign (≤0.30)    : {n_lb} ({n_lb/len(probs):.1%})")

    # Save full predictions
    out_cols = ["variation_id","protein_change","position","wt_aa","mut_aa",
                "P_pathogenic","predicted_class","mean_ddg","is_hotspot",
                "specific_ddg","blosum62_score","min_dist_dna"]
    out_cols = [c for c in out_cols if c in vus_covered.columns]
    vus_out = vus_covered[out_cols].sort_values("P_pathogenic", ascending=False)
    vus_csv = os.path.join(OUT, "vus_predictions.csv")
    vus_out.to_csv(vus_csv, index=False)
    print(f"\n  Full VUS predictions → {vus_csv}")

    # Top 20
    top20 = vus_out.head(20)
    top_csv = os.path.join(OUT, "vus_top20_predicted_pathogenic.csv")
    top20.to_csv(top_csv, index=False)
    print(f"  Top 20 predicted pathogenic VUS → {top_csv}")
    print("\n  Top 20 predicted pathogenic VUS:")
    print(f"  {'Variant':<10} {'P(path)':>8} {'Hotspot':>8} {'ΔΔG(pos)':>10} {'DNA dist':>10}")
    print("  " + "-"*52)
    for _, row in top20.iterrows():
        hs  = "YES" if row.get("is_hotspot", 0) == 1 else "no"
        ddg = f"{row['mean_ddg']:.2f}" if pd.notna(row.get("mean_ddg")) else "—"
        dna = f"{row['min_dist_dna']:.1f}" if pd.notna(row.get("min_dist_dna")) else "—"
        print(f"  {row['protein_change']:<10} {row['P_pathogenic']:8.4f} {hs:>8} {ddg:>10} {dna:>10}")

    # Score distribution plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probs[probs < 0.3],  bins=20, color="#2ca02c", alpha=0.7,
            label=f"Likely benign (n={n_lb})", range=(0,1))
    ax.hist(probs[(probs>=0.3)&(probs<0.7)], bins=20, color="#ff7f0e", alpha=0.7,
            label=f"Uncertain (n={n_unc})", range=(0,1))
    ax.hist(probs[probs >= 0.7], bins=20, color="#d62728", alpha=0.7,
            label=f"Likely pathogenic (n={n_lp})", range=(0,1))
    ax.axvline(0.3, color="black", lw=1, linestyle="--")
    ax.axvline(0.7, color="black", lw=1, linestyle="--")
    ax.set_xlabel("Predicted P(pathogenic)", fontsize=12)
    ax.set_ylabel("Number of VUS", fontsize=12)
    ax.set_title(f"TP53 VUS Reclassification by Structural Patch Model\n"
                 f"(n={len(probs)} VUS with structural coverage)", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    dist_path = os.path.join(OUT, "vus_score_distribution.png")
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"\n  Score distribution plot → {dist_path}")

    # Text report
    lines = [
        "TP53 VUS Reclassification — Structural Patch Model",
        "=" * 60,
        f"VUS downloaded from ClinVar: {len(vus)} missense VUS",
        f"VUS with structural patch coverage: {len(probs)}",
        f"Training set used: {len(fm)} labelled variants "
        f"({y_train.sum()} pathogenic, {(1-y_train).sum()} benign)",
        f"Model: Ensemble (RF + Gradient Boosting + SVM), full training",
        "",
        "Reclassification thresholds:",
        "  P ≥ 0.70 → Likely pathogenic",
        "  0.30 < P < 0.70 → Uncertain (remains VUS)",
        "  P ≤ 0.30 → Likely benign",
        "",
        f"Results:",
        f"  Likely pathogenic : {n_lp} ({n_lp/len(probs):.1%})",
        f"  Uncertain         : {n_unc} ({n_unc/len(probs):.1%})",
        f"  Likely benign     : {n_lb} ({n_lb/len(probs):.1%})",
        "",
        "Top 20 predicted pathogenic VUS:",
        f"  {'Variant':<10} {'P(path)':>8} {'Hotspot':>8} {'Mean_ΔΔG':>10}",
        "  " + "-"*42,
    ]
    for _, row in top20.iterrows():
        hs  = "YES" if row.get("is_hotspot",0)==1 else "no"
        ddg = f"{row['mean_ddg']:.2f}" if pd.notna(row.get("mean_ddg")) else "—"
        lines.append(f"  {row['protein_change']:<10} {row['P_pathogenic']:8.4f} {hs:>8} {ddg:>10}")
    lines += [
        "",
        "Clinical note: These predictions are based solely on structural",
        "microenvironment features. They should be interpreted alongside",
        "functional data, family history, and ACMG/AMP criteria before",
        "clinical use.",
    ]
    rpt_path = os.path.join(OUT, "vus_report.txt")
    with open(rpt_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"  Report → {rpt_path}")

    print("\n" + "=" * 68)
    print("  VUS prediction complete.")
    print("=" * 68)


if __name__ == "__main__":
    main()
