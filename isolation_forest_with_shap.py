import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── pip install shap if missing ──
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP not installed. Run: pip install shap")
    SHAP_AVAILABLE = False

# ══════════════════════════════════════════════════════
# CONFIG  ← only section you need to change
# ══════════════════════════════════════════════════════
INPUT_FILE   = "Raw_data_with_CVs.csv"
RESULTS_CSV  = "outlier_detection_results_with_CVs.csv"
PIVOT_CSV    = "isolation_forest_model_result_pivot.csv"
OUTPUT_EXCEL = "isolation_forest_results_and_pivot.xlsx"

# SHAP outputs
SHAP_SUMMARY_PLOT  = "shap_summary_plot.png"
SHAP_BOT_CSV       = "shap_bot_explanations.csv"

COLUMNS_USE = [
    "active_day_ct",
    "avg_mean_comm_len",
    "pattern_repeat_ratio",
    "total_comm_ct",
    "Avg_sim",
]

# Isolation Forest params
N_ESTIMATORS  = 300
RANDOM_STATE  = 42
CONTAMINATION = 0.01

USE_SCALING   = True
FLOAT_DECIMALS = 6

# BOT vs SUSPECT threshold on iforest_score
# More negative = stronger anomaly = more confident BOT
# Derived from your data: mean=-0.672, use mean as split point
BOT_SUSPECT_THRESHOLD = -0.68   # tune this: below → BOT, above → SUSPECT

# ══════════════════════════════════════════════════════
# 1) READ DATA
# ══════════════════════════════════════════════════════
df = pd.read_csv(INPUT_FILE)
df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()

missing = [c for c in COLUMNS_USE if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}\nAvailable: {df.columns.tolist()}")

for c in COLUMNS_USE:
    df[c] = pd.to_numeric(df[c], errors="coerce")

before    = len(df)
df_model  = df.dropna(subset=COLUMNS_USE).copy()
after     = len(df_model)
print(f"Rows before dropna: {before:,}")
print(f"Rows after  dropna: {after:,}")

# ══════════════════════════════════════════════════════
# 2) FIT ISOLATION FOREST
# ══════════════════════════════════════════════════════
X = df_model[COLUMNS_USE].values

if USE_SCALING:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X

iso_forest = IsolationForest(
    n_estimators  = N_ESTIMATORS,
    random_state  = RANDOM_STATE,
    contamination = CONTAMINATION,
)
iso_forest.fit(X_scaled)

# Predict
df_model["outlier"]       = iso_forest.predict(X_scaled)
df_model["iforest_score"] = iso_forest.score_samples(X_scaled)
df_model["Model_Result"]  = np.where(df_model["outlier"] == -1, "BOT", "NON BOT")

# ══════════════════════════════════════════════════════
# 3) BOT vs SUSPECT SPLIT  ← new
# ══════════════════════════════════════════════════════
# Among flagged BOTs, split by iforest_score strength
# More negative score = stronger anomaly = confirmed BOT
# Less negative (closer to 0) = weaker signal = SUSPECT

def assign_final_label(row):
    if row["Model_Result"] == "NON BOT":
        return "NON BOT"
    # It's a BOT — now decide BOT vs SUSPECT
    if row["iforest_score"] <= BOT_SUSPECT_THRESHOLD:
        return "BOT"
    else:
        return "SUSPECT"

df_model["Final_Label"] = df_model.apply(assign_final_label, axis=1)

print("\nLabel distribution:")
print(df_model["Final_Label"].value_counts().to_string())

# ══════════════════════════════════════════════════════
# 4) SHAP EXPLAINABILITY  ← new
# ══════════════════════════════════════════════════════
if SHAP_AVAILABLE:
    print("\nRunning SHAP explainability...")

    # TreeExplainer works directly with IsolationForest
    explainer   = shap.TreeExplainer(iso_forest)
    shap_values = explainer.shap_values(X_scaled)

    # shap_values shape: (n_rows, n_features)
    df_shap = pd.DataFrame(
        shap_values,
        columns=[f"shap_{c}" for c in COLUMNS_USE],
        index=df_model.index
    )

    # Add absolute SHAP values — tells you importance magnitude per feature
    for c in COLUMNS_USE:
        df_shap[f"shap_abs_{c}"] = df_shap[f"shap_{c}"].abs()

    # Top driving feature per UUID
    abs_cols = [f"shap_abs_{c}" for c in COLUMNS_USE]
    df_shap["top_feature"] = df_shap[abs_cols].idxmax(axis=1)\
                                               .str.replace("shap_abs_", "")
    df_shap["top_feature_shap"] = df_shap[abs_cols].max(axis=1).round(4)

    # Merge SHAP back into model df
    df_model = pd.concat([df_model.reset_index(drop=True),
                          df_shap.reset_index(drop=True)], axis=1)

    # ── SHAP Summary Plot (all data) ──────────────────
    print(f"  Saving SHAP summary plot → {SHAP_SUMMARY_PLOT}")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_scaled,
        feature_names=COLUMNS_USE,
        show=False
    )
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PLOT, dpi=150, bbox_inches="tight")
    plt.close()

    # ── SHAP detail for BOTs only ─────────────────────
    bot_mask      = df_model["Final_Label"].isin(["BOT", "SUSPECT"])
    bot_indices   = df_model[bot_mask].index
    shap_bots     = shap_values[df_model.index.get_indexer(bot_indices)]
    X_bots        = X_scaled[df_model.index.get_indexer(bot_indices)]

    # BOT-specific summary plot
    bot_plot_file = "shap_bot_only_plot.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_bots,
        X_bots,
        feature_names=COLUMNS_USE,
        show=False,
        title="SHAP Values — BOT + SUSPECT Users Only"
    )
    plt.tight_layout()
    plt.savefig(bot_plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saving BOT-only SHAP plot → {bot_plot_file}")

    # ── Per-UUID BOT explanation CSV ──────────────────
    bot_explain_cols = (
        ["uuid", "Final_Label", "iforest_score", "top_feature", "top_feature_shap"]
        + [f"shap_{c}" for c in COLUMNS_USE]
        + COLUMNS_USE
    )
    # keep only cols that exist
    bot_explain_cols = [c for c in bot_explain_cols if c in df_model.columns]

    df_bot_explain = df_model[df_model["Final_Label"].isin(["BOT","SUSPECT"])]\
                              [bot_explain_cols].copy()

    # Round SHAP cols to 4dp
    shap_cols = [c for c in df_bot_explain.columns if c.startswith("shap_")]
    df_bot_explain[shap_cols] = df_bot_explain[shap_cols].round(4)
    df_bot_explain["iforest_score"] = df_bot_explain["iforest_score"].round(4)

    df_bot_explain.to_csv(SHAP_BOT_CSV, index=False)
    print(f"  Saved per-UUID BOT explanations → {SHAP_BOT_CSV}")

    # ── Print top 10 BOTs with explanation ────────────
    print("\n── Top BOTs with SHAP explanation (sample) ──")
    print(df_bot_explain.sort_values("iforest_score")
                        .head(10)
                        [["uuid","Final_Label","iforest_score",
                          "top_feature","top_feature_shap"]]
                        .to_string(index=False))

    print("\nHow to read SHAP values:")
    print("  Negative SHAP → feature pushed the score toward anomaly (BOT)")
    print("  Positive SHAP → feature pushed the score toward normal (NON BOT)")
    print("  top_feature   → single biggest reason this UUID was flagged")

else:
    print("SHAP skipped — install with: pip install shap")

# ══════════════════════════════════════════════════════
# 5) SAVE RESULTS  (same as your original + new cols)
# ══════════════════════════════════════════════════════
# Columns to bring from model df into final output
merge_cols = ["uuid", "outlier", "Model_Result", "iforest_score", "Final_Label"]
if SHAP_AVAILABLE:
    merge_cols += [f"shap_{c}" for c in COLUMNS_USE]   # shap value per feature
    merge_cols += ["top_feature", "top_feature_shap"]   # top reason per UUID

# Keep only cols that exist in df_model
merge_cols = [c for c in merge_cols if c in df_model.columns]

df_out = df.merge(df_model[merge_cols], on="uuid", how="left")

# Round SHAP cols to 4dp in final output
if SHAP_AVAILABLE:
    shap_out_cols = [c for c in df_out.columns if c.startswith("shap_")]
    df_out[shap_out_cols] = df_out[shap_out_cols].round(4)

df_out.to_csv(RESULTS_CSV, index=False, float_format=f"%.{FLOAT_DECIMALS}f")
print(f"\n[DONE] Saved results → {RESULTS_CSV}")
print(f"       Columns in output: {df_out.columns.tolist()}")

# Pivot summary
pivot = df_out.groupby("Final_Label").agg(
    count        = ("uuid", "count"),
    mean_score   = ("iforest_score", "mean"),
    mean_prr     = ("pattern_repeat_ratio", "mean"),
    mean_comm_len= ("avg_mean_comm_len", "mean"),
).round(4)

pivot.to_csv(PIVOT_CSV)
print(f"[DONE] Saved pivot    → {PIVOT_CSV}")

if SHAP_AVAILABLE:
    print(f"[DONE] Saved SHAP explanations → {SHAP_BOT_CSV}")
    print(f"[DONE] Saved SHAP plots → {SHAP_SUMMARY_PLOT}, shap_bot_only_plot.png")
