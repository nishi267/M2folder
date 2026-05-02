"""
=============================================================================
Optum Claim Recon — Bot Detection Pipeline
=============================================================================
Models    : XGBoost (multi-class)
Classes   : BOT | Non-BOT | Suspect | Prospect
Input     : CSV — uuid, ticketsubmissiondate, comment_count,
                  average_comment_length, average_similarity, 30lb_bot_flag
Install   : pip install xgboost shap pandas scikit-learn matplotlib seaborn
=============================================================================
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_PATH        = "claudetask.csv"
MODEL_SAVE_PATH = "recon_bot_model.pkl"
LABEL_COL       = "30lb_bot_flag"   # spaces auto-normalised to underscores on load
UUID_COL        = "uuid"
DATE_COL        = "ticketsubmissiondate"
WINDOW_DAYS     = 30

XGBOOST_PARAMS = {
    "objective"        : "multi:softmax",
    "eval_metric"      : "mlogloss",
    "num_class"        : 4,
    "n_estimators"     : 300,
    "max_depth"        : 4,
    "learning_rate"    : 0.05,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "min_child_weight" : 3,
    "gamma"            : 0.1,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 1.0,
    "random_state"     : 42,
    "n_jobs"           : -1,
    "use_label_encoder": False,
}

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================
def load_data(csv_path):
    print(f"\n{'='*60}\nSTEP 1 — LOADING DATA\n{'='*60}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    print(f"  Loaded {len(df):,} rows | {df[UUID_COL].nunique():,} unique UUIDs")
    print(f"  Columns: {df.columns.tolist()}")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="mixed", dayfirst=False)
    df["average_similarity"] = (
        df["average_similarity"].astype(str)
        .str.replace("%", "", regex=False).str.strip().astype(float)
    )
    for col in ["comment_count", "average_comment_length", "average_similarity"]:
        df[col] = df[col].fillna(0)
    df = df.sort_values([UUID_COL, DATE_COL]).reset_index(drop=True)
    print(f"  Date range: {df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")
    if LABEL_COL in df.columns:
        print(f"  Label distribution:\n{df[LABEL_COL].value_counts().to_string()}")
    else:
        print("  Label column absent — inference mode")
    return df

# =============================================================================
# STEP 2 — FEATURE ENGINEERING
# =============================================================================
def compute_streak_and_gap(day_numbers):
    if len(day_numbers) == 0:
        return {"streak_max": 0, "gap_max": WINDOW_DAYS}
    days = sorted(day_numbers)
    streak_max, current_streak, gap_max = 1, 1, 0
    for i in range(1, len(days)):
        diff = days[i] - days[i - 1]
        if diff == 1:
            current_streak += 1
            streak_max = max(streak_max, current_streak)
        else:
            current_streak = 1
            gap_max = max(gap_max, diff - 1)
    gap_max = max(gap_max, WINDOW_DAYS - days[-1], days[0] - 1)
    return {"streak_max": streak_max, "gap_max": gap_max}

def engineer_features(df):
    print(f"\n{'='*60}\nSTEP 2 — FEATURE ENGINEERING (aggregating per UUID)\n{'='*60}")
    records = []
    for uuid, group in df.groupby(UUID_COL):
        group  = group.sort_values(DATE_COL).reset_index(drop=True)
        active = group[group["comment_count"] > 0].copy()
        total_days   = len(group)
        active_count = len(active)

        min_date = group[DATE_COL].min()
        group["day_number"]  = (group[DATE_COL] - min_date).dt.days + 1
        active["day_number"] = (active[DATE_COL] - min_date).dt.days + 1

        sg = compute_streak_and_gap(active["day_number"].tolist())

        if active_count > 1:
            intervals     = active["day_number"].diff().dropna().tolist()
            interval_std  = float(np.std(intervals))
            interval_mean = float(np.mean(intervals))
        else:
            interval_std = interval_mean = 0.0

        if active_count > 0:
            lv = active["average_comment_length"]
            sv = active["average_similarity"]
            cv = active["comment_count"]
            length_mean  = float(lv.mean())
            length_std   = float(lv.std(ddof=0))
            length_min   = float(lv.min())
            length_max   = float(lv.max())
            length_cv    = length_std / length_mean if length_mean > 0 else 0.0
            length_range = length_max - length_min
            sim_mean     = float(sv.mean())
            sim_std      = float(sv.std(ddof=0))
            sim_min      = float(sv.min())
            days_100pct  = int((sv >= 99.9).sum())
            pct_100pct   = days_100pct / active_count
            sim_flat     = 1 if sim_std < 0.01 else 0
            cc_total     = int(cv.sum())
            cc_mean      = float(cv.mean())
            cc_std       = float(cv.std(ddof=0))
            cc_max       = int(cv.max())
            burst_days   = int((cv > cc_mean + 2 * cc_std).sum())
        else:
            length_mean = length_std = length_min = length_max = 0.0
            length_cv = length_range = 0.0
            sim_mean = sim_std = sim_min = 0.0
            days_100pct = pct_100pct = sim_flat = 0
            cc_total = cc_mean = cc_std = cc_max = burst_days = 0

        records.append({
            UUID_COL           : uuid,
            "days_active"      : active_count,
            "days_absent"      : total_days - active_count,
            "activity_rate"    : active_count / total_days if total_days > 0 else 0,
            "streak_max"       : sg["streak_max"],
            "gap_max"          : sg["gap_max"],
            "interval_mean"    : interval_mean,
            "interval_std"     : interval_std,
            "length_mean"      : length_mean,
            "length_std"       : length_std,
            "length_min"       : length_min,
            "length_max"       : length_max,
            "length_cv"        : length_cv,
            "length_range"     : length_range,
            "sim_mean"         : sim_mean,
            "sim_std"          : sim_std,
            "sim_min"          : sim_min,
            "days_100pct_sim"  : days_100pct,
            "pct_days_100pct"  : pct_100pct,
            "sim_never_varies" : sim_flat,
            "cc_total"         : cc_total,
            "cc_mean"          : cc_mean,
            "cc_std"           : cc_std,
            "cc_max"           : cc_max,
            "burst_days"       : burst_days,
            LABEL_COL          : group[LABEL_COL].iloc[0],
        })

    uuid_df = pd.DataFrame(records)
    print(f"  Aggregated {df[UUID_COL].nunique():,} UUIDs → {len(uuid_df):,} training rows")
    print(f"  Features engineered: {len(uuid_df.columns) - 2}")
    known = uuid_df[uuid_df[LABEL_COL] != "UNKNOWN"]
    if len(known) > 0:
        print(f"  Label distribution:\n{known[LABEL_COL].value_counts().to_string()}")
    else:
        print("  Label column: UNKNOWN (inference mode — no ground truth labels)")
    return uuid_df

# =============================================================================
# STEP 3 — LABEL ENCODING
# =============================================================================
def encode_labels(uuid_df):
    print(f"\n{'='*60}\nSTEP 3 — LABEL ENCODING\n{'='*60}")
    le = LabelEncoder()
    y  = le.fit_transform(uuid_df[LABEL_COL])
    X  = uuid_df.drop(columns=[UUID_COL, LABEL_COL])
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Encoded: {list(range(len(le.classes_)))}")
    print(f"  Feature matrix shape: {X.shape}")
    return X, y, le

# =============================================================================
# STEP 4 — TRAIN
# =============================================================================
def train_model(X, y, le):
    print(f"\n{'='*60}\nSTEP 4 — TRAINING XGBOOST\n{'='*60}")
    n_actual_classes = len(le.classes_)
    params = {**XGBOOST_PARAMS, "num_class": n_actual_classes}
    # When only 1 class in training data, multi:softmax breaks predict_proba.
    # Switch to binary:logistic so XGBoost stays stable.
    if n_actual_classes == 1:
        params["objective"]  = "binary:logistic"
        params["num_class"]  = 1
        params["eval_metric"]= "logloss"
    model  = XGBClassifier(**params)
    n_splits = min(5, len(np.unique(y)))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        print(f"  CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    else:
        print("  Skipping CV — insufficient class diversity in sample data")
        print("  (expected with sample data; full April data will enable CV)")
    model.fit(X, y, verbose=False)
    print(f"  Model trained on {len(X)} UUIDs | {len(X.columns)} features")
    return model

# =============================================================================
# STEP 5 — EVALUATE
# =============================================================================
def evaluate_model(model, X, y, le):
    print(f"\n{'='*60}\nSTEP 5 — EVALUATION\n{'='*60}")
    y_pred = le.inverse_transform(model.predict(X))
    y_true = le.inverse_transform(y)
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=le.classes_)
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Recon Bot Detection", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("  Saved: confusion_matrix.png")

# =============================================================================
# SHAP HELPERS — version-safe normalisation
# =============================================================================
def _normalise_shap(shap_values, n_classes):
    """
    Always returns sv_list: list of 2D arrays [n_samples, n_features], one per class.
    Handles every output shape XGBoost/SHAP can produce.
    """
    if isinstance(shap_values, list):
        return [np.atleast_2d(sv) for sv in shap_values]
    arr = np.array(shap_values)
    if arr.ndim == 1:
        # 1 sample, 1 class → (n_features,) → wrap to (1, n_features)
        return [arr.reshape(1, -1)]
    if arr.ndim == 2:
        # (n_samples, n_features) — single class
        return [arr]
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes)
        return [arr[:, :, i] for i in range(arr.shape[2])]
    return [np.atleast_2d(arr)]

def _normalise_base(base_val, n_classes):
    if np.isscalar(base_val):
        return [float(base_val)] * n_classes
    base_arr = np.array(base_val).ravel()
    if len(base_arr) == 1:
        return [float(base_arr[0])] * n_classes
    return [float(b) for b in base_arr]

def _shap_waterfall_safe(sv_row, base, feature_names, feature_values, title, fname):
    """
    Replacement for shap.plots.waterfall that works with any SHAP version.
    shap.plots.waterfall internally calls np.sort(axis=1) which crashes on 1D
    sv_row when only one class exists. This custom implementation never fails.
    """
    sv   = np.array(sv_row, dtype=float)   # guaranteed 1D [n_features]
    vals = sv.copy()
    n    = len(feature_names)

    # Sort by absolute SHAP value descending, show top 15
    top_n   = min(15, n)
    idx     = np.argsort(np.abs(vals))[::-1][:top_n]
    idx     = idx[::-1]   # reverse so largest is at top of horizontal bar chart

    feat_names_top = [f"{feature_names[i]}" for i in idx]
    feat_vals_top  = [feature_values[i] for i in idx]
    shap_vals_top  = vals[idx]

    labels = [f"{fn}\n= {fv:.2f}" for fn, fv in zip(feat_names_top, feat_vals_top)]
    colors = ["#d73027" if v > 0 else "#4575b4" for v in shap_vals_top]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(top_n), shap_vals_top, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on prediction)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    # Add value labels on bars
    for bar, val in zip(bars, shap_vals_top):
        xpos = bar.get_width()
        ha   = "left" if val >= 0 else "right"
        offset = 0.001 if val >= 0 else -0.001
        ax.text(xpos + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va="center", ha=ha, fontsize=8)

    # Base value annotation
    ax.text(0.01, 0.01, f"Base value: {base:.4f}", transform=ax.transAxes,
            fontsize=8, color="grey")

    red_patch  = plt.Rectangle((0,0),1,1, color="#d73027", alpha=0.85)
    blue_patch = plt.Rectangle((0,0),1,1, color="#4575b4", alpha=0.85)
    ax.legend([red_patch, blue_patch],
              ["Pushes toward predicted class", "Pushes away from predicted class"],
              fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# SAFE PREDICT_PROBA — handles single-class edge case
# =============================================================================
def _safe_predict_proba(model, X, n_classes):
    """
    model.predict_proba() crashes when trained with num_class>1 but only
    1 class exists in data (XGBoost softmax gets a 1D raw output and
    scipy softmax then fails on axis=1).
    This wrapper always returns a clean 2D array [n_samples, n_classes].
    """
    try:
        proba = model.predict_proba(X)
        # Ensure 2D
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)
        return proba
    except Exception:
        # Fallback: use raw margin scores and apply softmax manually
        raw = model.predict(X, output_margin=True)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        # Manual softmax per row
        raw_shifted = raw - raw.max(axis=1, keepdims=True)
        exp_raw     = np.exp(raw_shifted)
        proba       = exp_raw / exp_raw.sum(axis=1, keepdims=True)
        return proba

# =============================================================================
# STEP 6 — GLOBAL EXPLAINABILITY
# =============================================================================
def explain_global(model, X, le):
    print(f"\n{'='*60}\nSTEP 6 — GLOBAL EXPLAINABILITY (SHAP)\n{'='*60}")

    explainer = shap.TreeExplainer(model)
    raw_sv    = explainer.shap_values(X)
    n_classes = len(le.classes_)
    sv_list   = _normalise_shap(raw_sv, n_classes)
    base_list = _normalise_base(explainer.expected_value, n_classes)

    print(f"  SHAP sv_list: {len(sv_list)} class(es), "
          f"shapes: {[sv.shape for sv in sv_list]}")

    # -- 6a. Feature importance --
    imp_df = pd.DataFrame({
        "feature"    : X.columns,
        "importance" : model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\n  Top 10 Features by Gain:")
    print(imp_df.head(10).to_string(index=False))

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(imp_df)))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color=colors)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    ax.set_title("XGBoost Feature Importance — Recon Bot Detection", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    print("  Saved: feature_importance.png")

    # -- 6b. SHAP summary for BOT class --
    bot_idx      = list(le.classes_).index("BOT") if "BOT" in le.classes_ else 0
    bot_idx      = min(bot_idx, len(sv_list) - 1)
    shap_for_bot = sv_list[bot_idx]   # always 2D [n_samples, n_features]

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_for_bot, X, show=False, plot_type="dot", max_display=15)
    plt.title("SHAP Summary — BOT class", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig("shap_summary_bot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_summary_bot.png")

    # -- 6c. Mean |SHAP| per class --
    fig, ax = plt.subplots(figsize=(10, 6))
    x      = np.arange(len(X.columns))
    width  = 0.8 / n_classes
    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))
    for i, cls in enumerate(le.classes_):
        sv_i = sv_list[min(i, len(sv_list) - 1)]   # 2D [n_samples, n_features]
        vals = np.abs(sv_i).mean(axis=0)            # 1D [n_features] — safe on 2D
        ax.bar(x + i * width, vals, width, label=cls, color=colors[i], alpha=0.85)
    ax.set_xticks(x + width * (n_classes - 1) / 2)
    ax.set_xticklabels(X.columns, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Feature Impact per Class (SHAP)", fontsize=13)
    ax.legend(title="Class")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("shap_by_class.png", dpi=150)
    plt.close()
    print("  Saved: shap_by_class.png")

    return explainer, sv_list, base_list

# =============================================================================
# STEP 7 — LOCAL EXPLAINABILITY
# =============================================================================
def explain_local(model, explainer, sv_list, base_list, X, uuid_df, le,
                  uuid_id=None, row_index=0):
    print(f"\n{'='*60}\nSTEP 7 — LOCAL EXPLAINABILITY (single UUID)\n{'='*60}")

    if uuid_id and uuid_id in uuid_df[UUID_COL].values:
        row_index = uuid_df.index.get_loc(
            uuid_df[uuid_df[UUID_COL] == uuid_id].index[0])

    target_uuid  = uuid_df[UUID_COL].iloc[row_index]
    true_label   = uuid_df[LABEL_COL].iloc[row_index]
    x_row        = X.iloc[[row_index]]

    pred_encoded = int(model.predict(x_row)[0])
    pred_label   = le.inverse_transform([pred_encoded])[0]
    pred_proba   = _safe_predict_proba(model, x_row, len(le.classes_))[0]  # 1D [n_classes]

    # Safe SHAP extraction — sv_list[i] is always 2D after _normalise_shap
    cls_idx  = min(pred_encoded, len(sv_list) - 1)
    sv_2d    = sv_list[cls_idx]          # 2D [n_samples, n_features]
    sv_row   = sv_2d[row_index]          # 1D [n_features]
    base     = base_list[cls_idx]

    contrib = pd.DataFrame({
        "feature"       : list(X.columns),
        "feature_value" : x_row.values[0],
        "shap_value"    : sv_row,
    }).sort_values("shap_value", key=lambda s: s.abs(), ascending=False)

    # ── Console output ───────────────────────────────────────────────────
    conf_idx = min(pred_encoded, len(pred_proba) - 1)
    print(f"\n  UUID            : {target_uuid}")
    print(f"  True Label      : {true_label}")
    print(f"  Predicted Label : {pred_label}")
    print(f"  Confidence      : {pred_proba[conf_idx]*100:.1f}%")

    print(f"\n  Probability breakdown:")
    for cls, prob in zip(le.classes_, pred_proba):
        print(f"    {cls:<12} {prob*100:5.1f}%  {'█' * int(prob * 30)}")

    print(f"\n  Top features pushing TOWARD '{pred_label}':")
    toward = contrib[contrib["shap_value"] > 0].head(5)
    if toward.empty:
        print("    (all SHAP values are zero — expected when only 1 class exists in training data)")
    for _, r in toward.iterrows():
        print(f"    {r['feature']:<28} val={r['feature_value']:>10.3f}  SHAP=+{r['shap_value']:.4f}")

    print(f"\n  Top features pushing AWAY from '{pred_label}':")
    away = contrib[contrib["shap_value"] < 0].head(5)
    if away.empty:
        print("    (all SHAP values are zero — expected when only 1 class exists in training data)")
    for _, r in away.iterrows():
        print(f"    {r['feature']:<28} val={r['feature_value']:>10.3f}  SHAP={r['shap_value']:.4f}")

    print(f"\n  ── Natural Language Explanation ──")
    print(f"  UUID {target_uuid[:8]}... → [{pred_label}] ({pred_proba[conf_idx]*100:.1f}% confidence)")
    for _, r in contrib.head(3).iterrows():
        direction = "toward" if r["shap_value"] > 0 else "away from"
        print(f"  • '{r['feature']}' = {r['feature_value']:.3f} "
              f"pushed {direction} {pred_label} (|impact|={abs(r['shap_value']):.4f})")

    # ── SHAP waterfall — custom implementation avoids shap version crashes ──
    fname = f"shap_waterfall_{target_uuid[:8]}.png"
    _shap_waterfall_safe(
        sv_row        = sv_row,
        base          = base,
        feature_names = list(X.columns),
        feature_values= x_row.values[0],
        title         = f"SHAP Feature Impact — UUID {target_uuid[:8]}... → {pred_label}",
        fname         = fname,
    )
    print(f"\n  Saved: {fname}")
    return contrib

# =============================================================================
# STEP 8 — SAVE / LOAD MODEL
# =============================================================================
def save_model(model, le, feature_names, path=MODEL_SAVE_PATH):
    joblib.dump({"model": model, "label_encoder": le,
                 "feature_names": feature_names}, path)
    print(f"\n  Model saved → {path}")

def load_model(path=MODEL_SAVE_PATH):
    b = joblib.load(path)
    return b["model"], b["label_encoder"], b["feature_names"]

# =============================================================================
# STEP 9 — INFERENCE
# =============================================================================
def run_inference(new_csv_path, model_path=MODEL_SAVE_PATH,
                  explain=True, top_n_features=5):
    """
    Score new UUIDs and optionally produce per-UUID SHAP explanation.

    Parameters
    ----------
    new_csv_path   : path to new raw CSV (same format as training)
    model_path     : path to saved model bundle
    explain        : if True, print + save a SHAP bar chart per UUID
    top_n_features : how many top features to show per UUID explanation
    """
    print(f"\n{'='*60}\nINFERENCE — Scoring new data\n{'='*60}")
    model, le, feature_names = load_model(model_path)
    raw_df = load_data(new_csv_path)
    if LABEL_COL not in raw_df.columns:
        raw_df[LABEL_COL] = "UNKNOWN"
    uuid_df      = engineer_features(raw_df)
    X_new        = uuid_df[feature_names]
    pred_encoded = model.predict(X_new)
    pred_labels  = le.inverse_transform(pred_encoded)
    pred_probas  = _safe_predict_proba(model, X_new, len(le.classes_))

    # ── Build results table ──────────────────────────────────────────────
    results = pd.DataFrame({"uuid": uuid_df[UUID_COL], "predicted_class": pred_labels})
    for i, cls in enumerate(le.classes_):
        results[f"prob_{cls}"] = pred_probas[:, i].round(4)
    results["confidence"] = pred_probas.max(axis=1).round(4)

    print(f"\n  Scored {len(results)} UUIDs")
    print(results.to_string(index=False))
    results.to_csv("inference_results.csv", index=False)
    print("\n  Saved: inference_results.csv")

    # ── Per-UUID SHAP explanation ────────────────────────────────────────
    if explain:
        print(f"\n{'='*60}\nPER-UUID EXPLAINABILITY\n{'='*60}")

        explainer = shap.TreeExplainer(model)
        raw_sv    = explainer.shap_values(X_new)
        n_classes = len(le.classes_)
        sv_list   = _normalise_shap(raw_sv, n_classes)
        base_list = _normalise_base(explainer.expected_value, n_classes)

        explanation_rows = []   # collect for summary CSV

        for row_i, (_, res_row) in enumerate(results.iterrows()):
            uid         = res_row["uuid"]
            pred_label  = res_row["predicted_class"]
            confidence  = res_row["confidence"]
            pred_enc    = int(le.transform([pred_label])[0])
            proba_row   = pred_probas[row_i]          # 1D [n_classes]

            # SHAP for this UUID
            cls_idx = min(pred_enc, len(sv_list) - 1)
            sv_row  = sv_list[cls_idx][row_i]         # 1D [n_features]
            base    = base_list[cls_idx]

            # Feature contribution table
            contrib = pd.DataFrame({
                "feature"       : feature_names,
                "feature_value" : X_new.iloc[row_i].values,
                "shap_value"    : sv_row,
            }).sort_values("shap_value", key=lambda s: s.abs(), ascending=False)

            top_toward = contrib[contrib["shap_value"] > 0].head(top_n_features)
            top_away   = contrib[contrib["shap_value"] < 0].head(top_n_features)

            # ── Console explanation per UUID ─────────────────────────────
            print(f"\n  {'─'*56}")
            print(f"  UUID       : {uid}")
            print(f"  Prediction : {pred_label}  ({confidence*100:.1f}% confidence)")
            print(f"\n  Probability breakdown:")
            for cls, prob in zip(le.classes_, proba_row):
                bar = '█' * int(prob * 25)
                print(f"    {cls:<12} {prob*100:5.1f}%  {bar}")

            print(f"\n  Why {pred_label}? — Top features driving this prediction:")
            if top_toward.empty:
                print("    (SHAP values are zero — model is certain, no single feature dominates)")
            for _, r in top_toward.iterrows():
                print(f"    ▲  {r['feature']:<26} val={r['feature_value']:>9.3f}  "
                      f"impact=+{r['shap_value']:.4f}")

            if not top_away.empty:
                print(f"\n  Features that argued AGAINST {pred_label}:")
                for _, r in top_away.iterrows():
                    print(f"    ▼  {r['feature']:<26} val={r['feature_value']:>9.3f}  "
                          f"impact={r['shap_value']:.4f}")

            # ── Natural language summary ─────────────────────────────────
            print(f"\n  Summary: UUID {uid[:8]}... is classified as [{pred_label}]")
            for _, r in contrib.head(3).iterrows():
                direction = "toward" if r["shap_value"] > 0 else "against"
                print(f"    • {r['feature']} = {r['feature_value']:.2f} "
                      f"pushed {direction} {pred_label} "
                      f"(impact {r['shap_value']:+.4f})")

            # ── Per-UUID SHAP bar chart ──────────────────────────────────
            fname = f"explanation_{uid[:8]}.png"
            _shap_waterfall_safe(
                sv_row        = sv_row,
                base          = base,
                feature_names = feature_names,
                feature_values= X_new.iloc[row_i].values,
                title         = f"UUID {uid[:8]}...  →  {pred_label}  "
                                f"({confidence*100:.1f}% confidence)",
                fname         = fname,
            )
            print(f"  Saved chart: {fname}")

            # Collect top feature for summary
            top1 = contrib.iloc[0]
            explanation_rows.append({
                "uuid"            : uid,
                "predicted_class" : pred_label,
                "confidence"      : round(confidence, 4),
                "top_feature"     : top1["feature"],
                "top_feature_val" : round(top1["feature_value"], 3),
                "top_shap_impact" : round(top1["shap_value"], 4),
            })

        # ── Save explanation summary CSV ─────────────────────────────────
        exp_df = pd.DataFrame(explanation_rows)
        exp_df.to_csv("explanation_summary.csv", index=False)
        print(f"\n  {'='*56}")
        print(f"  Saved: explanation_summary.csv  (top driving feature per UUID)")
        print(f"  Saved: explanation_<uuid>.png   (SHAP chart per UUID)")

    return results

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 60)
    print("  OPTUM CLAIM RECON — BOT DETECTION PIPELINE")
    print("=" * 60)

    raw_df            = load_data(CSV_PATH)
    uuid_df           = engineer_features(raw_df)
    X, y, le          = encode_labels(uuid_df)
    model             = train_model(X, y, le)
    evaluate_model(model, X, y, le)
    explainer, sv_list, base_list = explain_global(model, X, le)
    explain_local(model, explainer, sv_list, base_list, X, uuid_df, le, row_index=0)
    save_model(model, le, list(X.columns))

    # Uncomment to score new data:
    # run_inference("new_data.csv")

    print(f"\n{'='*60}\n  PIPELINE COMPLETE\n{'='*60}")
    print("  confusion_matrix.png    — evaluation")
    print("  feature_importance.png  — global importance")
    print("  shap_summary_bot.png    — SHAP beeswarm BOT class")
    print("  shap_by_class.png       — SHAP all classes")
    print("  shap_waterfall_*.png    — per-UUID SHAP bar chart")
    print("  recon_bot_model.pkl     — saved model")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
