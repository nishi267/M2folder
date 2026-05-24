import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP not installed. Run: pip install shap")
    SHAP_AVAILABLE = False

# ══════════════════════════════════════════════════════
# CONFIG  ← only change this section
# ══════════════════════════════════════════════════════
INPUT_FILE   = "Raw_data_with_CVs.csv"
RESULTS_CSV  = "outlier_detection_results_with_CVs.csv"
PIVOT_CSV    = "isolation_forest_model_result_pivot.csv"

SHAP_SUMMARY_PLOT = "shap_summary_plot.png"
SHAP_BOT_PLOT     = "shap_bot_only_plot.png"

COLUMNS_USE = [
    "active_day_ct",
    "avg_mean_comm_len",
    "pattern_repeat_ratio",
    "total_comm_ct",
    "Avg_sim",
]

N_ESTIMATORS  = 300
RANDOM_STATE  = 42
CONTAMINATION = 0.01
USE_SCALING   = True
FLOAT_DECIMALS = 6

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

before   = len(df)
df_model = df.dropna(subset=COLUMNS_USE).copy()
after    = len(df_model)
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

# outlier: -1 = outlier (BOT), 1 = inlier (NON BOT)
df_model["outlier"]       = iso_forest.predict(X_scaled)
df_model["iforest_score"] = iso_forest.score_samples(X_scaled)
df_model["Model_Result"]  = np.where(df_model["outlier"] == -1, "BOT", "NON BOT")

# ══════════════════════════════════════════════════════
# 3) SHAP
# ══════════════════════════════════════════════════════
if SHAP_AVAILABLE:
    print("\nRunning SHAP...")
    explainer   = shap.TreeExplainer(iso_forest)
    shap_values = explainer.shap_values(X_scaled)

    # One SHAP column per feature for every row
    df_shap = pd.DataFrame(
        shap_values,
        columns=[f"shap_{c}" for c in COLUMNS_USE],
        index=df_model.index
    ).round(4)

    # Top driving feature per row
    abs_cols = df_shap.abs()
    df_shap["top_feature"]      = abs_cols.idxmax(axis=1).str.replace("shap_", "")
    df_shap["top_feature_shap"] = abs_cols.max(axis=1).round(4)

    # Merge SHAP into model df
    df_model = pd.concat([
        df_model.reset_index(drop=True),
        df_shap.reset_index(drop=True)
    ], axis=1)
    print(f"  SHAP values added to all {len(df_model)} rows")

    # Summary plot — all rows
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_scaled,
                      feature_names=COLUMNS_USE, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {SHAP_SUMMARY_PLOT}")

    # BOT-only plot
    bot_idx    = df_model[df_model["outlier"] == -1].index
    pos        = [df_model.index.get_loc(i) for i in bot_idx]
    shap_bots  = shap_values[pos]
    X_bots     = X_scaled[pos]
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_bots, X_bots,
                      feature_names=COLUMNS_USE, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_BOT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {SHAP_BOT_PLOT}")

# ══════════════════════════════════════════════════════
# 4) MERGE BACK TO ORIGINAL AND SAVE
# ══════════════════════════════════════════════════════
merge_cols = ["uuid", "outlier", "iforest_score", "Model_Result"]
if SHAP_AVAILABLE:
    merge_cols += [f"shap_{c}" for c in COLUMNS_USE]
    merge_cols += ["top_feature", "top_feature_shap"]

merge_cols = [c for c in merge_cols if c in df_model.columns]

df_out = df.merge(df_model[merge_cols], on="uuid", how="left")

df_out.to_csv(RESULTS_CSV, index=False, float_format=f"%.{FLOAT_DECIMALS}f")
print(f"\n[DONE] Saved → {RESULTS_CSV}")
print(f"       Rows: {len(df_out)}  |  Columns: {df_out.columns.tolist()}")

# Pivot
pivot = df_out.groupby("Model_Result").agg(
    count      = ("uuid",          "count"),
    mean_score = ("iforest_score", "mean"),
).round(4)
pivot.to_csv(PIVOT_CSV)
print(f"[DONE] Saved pivot → {PIVOT_CSV}")
