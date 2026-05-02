"""
=============================================================================
Optum Claim Recon — Vision CNN Pipeline
=============================================================================
Approach  : Convert each UUID's 30-day matrix → grayscale image → CNN classify
Model     : Custom lightweight 2D CNN (no pretrained weights needed)
Classes   : BOT | Non-BOT | Suspect | Prospect
Explain   : Grad-CAM per UUID — highlights which days × features drove prediction
Input     : Same CSV format as XGBoost pipeline
Install   : pip install torch torchvision pandas scikit-learn matplotlib pillow
=============================================================================

WHY VISION CNN FOR THIS DATA?
------------------------------
Each UUID has a 30-day × 3-feature matrix:
    columns = [comment_count, average_comment_length, average_similarity]
    rows    = Day 1 ... Day 30

When rendered as a grayscale image:
    BOT     → regular bright rows with almost no variation — striped pattern
    Non-BOT → noisy, irregular, high variance across rows and columns
    Suspect → semi-regular, moderate brightness
    Prospect → sparse, low brightness, few active rows

A 2D CNN's filters learn to detect these visual texture patterns
exactly the same way they detect edges or textures in photos.

GRAD-CAM EXPLAINABILITY:
    Produces a heatmap overlaid on the UUID's image showing
    WHICH DAYS × WHICH FEATURES drove the prediction.
    Red = high importance, Blue = low importance.
    Directly interpretable: "Model looked at days 5-10 similarity column"
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

from PIL import Image
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_PATH        = "claudetask_dummy.csv"
MODEL_SAVE_PATH = "recon_vision_cnn.pt"
LABEL_COL       = "30lb_bot_flag"
UUID_COL        = "uuid"
DATE_COL        = "ticketsubmissiondate"
WINDOW_DAYS     = 30
FEATURE_COLS    = ["comment_count", "average_comment_length", "average_similarity"]
N_FEATURES      = len(FEATURE_COLS)    # 3
IMAGE_H         = WINDOW_DAYS          # 30 rows  (days)
IMAGE_W         = N_FEATURES           # 3  cols  (features)
N_CLASSES       = 4
BATCH_SIZE      = 16
EPOCHS          = 50
LEARNING_RATE   = 1e-3
RANDOM_SEED     = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =============================================================================
# STEP 1 — LOAD AND PREPROCESS DATA
# =============================================================================
def load_data(csv_path):
    print(f"\n{'='*60}\nSTEP 1 — LOADING DATA\n{'='*60}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    print(f"  Loaded {len(df):,} rows | {df[UUID_COL].nunique():,} unique UUIDs")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="mixed", dayfirst=False)
    df["average_similarity"] = (
        df["average_similarity"].astype(str)
        .str.replace("%", "", regex=False).str.strip().astype(float)
    )
    for col in FEATURE_COLS:
        df[col] = df[col].fillna(0)

    df = df.sort_values([UUID_COL, DATE_COL]).reset_index(drop=True)
    if LABEL_COL in df.columns:
        print(f"  Label distribution:\n{df[LABEL_COL].value_counts().to_string()}")
    else:
        print("  Label column absent — inference mode")
    return df

# =============================================================================
# STEP 2 — BUILD IMAGE MATRIX PER UUID
# =============================================================================
def uuid_to_matrix(group):
    """
    Convert one UUID's 30-day data into a [30 × 3] float matrix.
    Each row = one day, each column = one feature.
    Absent days are already zero-filled.
    Returns matrix shape [WINDOW_DAYS, N_FEATURES].
    """
    group = group.sort_values(DATE_COL).reset_index(drop=True)

    # Build dense 30-row matrix — fill missing dates with zeros
    min_date = group[DATE_COL].min()
    dense = pd.DataFrame({"day_offset": range(WINDOW_DAYS)})
    group["day_offset"] = (group[DATE_COL] - min_date).dt.days
    merged = dense.merge(group[["day_offset"] + FEATURE_COLS], on="day_offset", how="left")
    merged[FEATURE_COLS] = merged[FEATURE_COLS].fillna(0)

    matrix = merged[FEATURE_COLS].values.astype(np.float32)   # [30, 3]
    return matrix

def normalise_matrix(matrix, feature_stats):
    """
    Min-max normalise each feature column to [0, 1] using training stats.
    feature_stats: dict of {feature_name: (min, max)}
    """
    normed = matrix.copy()
    for i, feat in enumerate(FEATURE_COLS):
        fmin, fmax = feature_stats[feat]
        denom = fmax - fmin if fmax > fmin else 1.0
        normed[:, i] = (matrix[:, i] - fmin) / denom
    return np.clip(normed, 0, 1)

def compute_feature_stats(df):
    """Compute per-feature min/max from training data for normalisation."""
    stats = {}
    for feat in FEATURE_COLS:
        stats[feat] = (df[feat].min(), df[feat].max())
    return stats

def build_image_dataset(df, feature_stats, label_encoder=None):
    """
    Build image tensors and labels from raw DataFrame.
    Returns:
        images  : np.array [n_uuids, 1, 30, 3] — channel-first for PyTorch
        labels  : np.array [n_uuids] int encoded (or None for inference)
        uuid_ids: list of UUID strings
        matrices: list of raw [30,3] matrices (for visualisation)
    """
    print(f"\n{'='*60}\nSTEP 2 — BUILDING IMAGE MATRICES\n{'='*60}")
    images, labels, uuid_ids, matrices = [], [], [], []

    has_label = LABEL_COL in df.columns

    for uid, group in df.groupby(UUID_COL):
        matrix = uuid_to_matrix(group)
        normed = normalise_matrix(matrix, feature_stats)

        # Shape [1, 30, 3] — single channel grayscale
        img_tensor = normed[np.newaxis, :, :]

        images.append(img_tensor)
        uuid_ids.append(uid)
        matrices.append(matrix)

        if has_label and label_encoder is not None:
            lbl = group[LABEL_COL].iloc[0]
            labels.append(label_encoder.transform([lbl])[0])
        elif has_label:
            labels.append(group[LABEL_COL].iloc[0])

    images = np.array(images, dtype=np.float32)   # [N, 1, 30, 3]
    labels = np.array(labels) if labels else None

    print(f"  Built {len(images)} image matrices, shape per image: {images[0].shape}")
    return images, labels, uuid_ids, matrices

# =============================================================================
# STEP 3 — PYTORCH DATASET
# =============================================================================
class ReconImageDataset(Dataset):
    def __init__(self, images, labels=None):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.labels is not None:
            return img, self.labels[idx]
        return img

# =============================================================================
# STEP 4 — CNN MODEL ARCHITECTURE
# =============================================================================
class ReconVisionCNN(nn.Module):
    """
    Lightweight custom 2D CNN for 30×3 grayscale images.
    Architecture designed for small spatial dimensions:
      - Input: [batch, 1, 30, 3]
      - Kernel sizes kept small to avoid spatial collapse
      - GlobalAvgPool replaces fully-connected flattening
      - Dropout for regularisation on small datasets
    """
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()

        # Block 1: detect local day patterns (kernel spans 3 days × 1 feature)
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32,
            kernel_size=(3, 1), padding=(1, 0)   # same height padding
        )
        self.bn1   = nn.BatchNorm2d(32)

        # Block 2: detect cross-feature patterns (kernel spans 1 day × all 3 features)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64,
            kernel_size=(3, 3), padding=(1, 1)
        )
        self.bn2   = nn.BatchNorm2d(64)

        # Block 3: detect longer temporal motifs (5-day windows)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128,
            kernel_size=(5, 1), padding=(2, 0)
        )
        self.bn3   = nn.BatchNorm2d(128)

        # Global average pool — collapses spatial dims to 1×1
        # Works regardless of input spatial size — no hardcoded flatten size
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.drop   = nn.Dropout(0.4)

        # Classification head
        self.fc     = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: [batch, 1, 30, 3]
        x = F.relu(self.bn1(self.conv1(x)))    # [batch, 32, 30, 3]
        x = F.relu(self.bn2(self.conv2(x)))    # [batch, 64, 30, 3]
        x = F.relu(self.bn3(self.conv3(x)))    # [batch, 128, 30, 3]
        x = self.gap(x)                        # [batch, 128, 1, 1]
        x = self.drop(x.view(x.size(0), -1))  # [batch, 128]
        return self.fc(x)                      # [batch, n_classes]

# =============================================================================
# STEP 5 — TRAINING
# =============================================================================
def train_model(images, labels, le, n_classes):
    print(f"\n{'='*60}\nSTEP 5 — TRAINING VISION CNN\n{'='*60}")

    dataset   = ReconImageDataset(images, labels)
    n_splits  = min(5, len(np.unique(labels)))

    if n_splits < 2:
        print("  Skipping CV — insufficient class diversity")
        # Train on full data
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        model  = _train_single(loader, n_classes)
    else:
        # Stratified K-Fold CV to report accuracy
        skf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        fold_accs = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
            train_ds  = ReconImageDataset(images[train_idx], labels[train_idx])
            val_ds    = ReconImageDataset(images[val_idx],   labels[val_idx])
            train_ldr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_ldr   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

            fold_model = _train_single(train_ldr, n_classes, verbose=False)
            acc = _evaluate_accuracy(fold_model, val_ldr)
            fold_accs.append(acc)
            print(f"  Fold {fold+1}/{n_splits} — val accuracy: {acc:.4f}")

        print(f"  CV accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")

        # Final model trained on all data
        full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        model = _train_single(full_loader, n_classes, verbose=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    return model

def _train_single(loader, n_classes, verbose=True):
    model     = ReconVisionCNN(n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch in loader:
            imgs, lbls = batch
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{EPOCHS}  loss={epoch_loss/len(loader):.4f}")

    return model

def _evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            imgs, lbls = batch
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total   += len(lbls)
    model.train()
    return correct / total if total > 0 else 0

# =============================================================================
# STEP 6 — EVALUATE
# =============================================================================
def evaluate_model(model, images, labels, le):
    print(f"\n{'='*60}\nSTEP 6 — EVALUATION\n{'='*60}")
    dataset = ReconImageDataset(images, labels)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE)

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            imgs, lbls = batch
            preds = model(imgs.to(DEVICE)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(lbls.numpy())

    y_pred = le.inverse_transform(all_preds)
    y_true = le.inverse_transform(all_true)

    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=le.classes_)
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Vision CNN", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig("cnn_confusion_matrix.png", dpi=150)
    plt.close()
    print("  Saved: cnn_confusion_matrix.png")

# =============================================================================
# GRAD-CAM — per-UUID visual explainability
# =============================================================================
class GradCAM:
    """
    Grad-CAM for the Vision CNN.
    Hooks into the last conv layer (conv3) to capture:
      - activations (forward pass feature maps)
      - gradients (backward pass gradients w.r.t. predicted class)
    Produces a heatmap [30 × 3] showing which day-feature cells
    most influenced the prediction.
    """
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def compute(self, img_tensor, class_idx):
        """
        img_tensor : [1, 1, 30, 3] torch tensor
        class_idx  : integer class to explain
        Returns heatmap [30, 3] normalised to [0, 1]
        """
        self.model.eval()
        img_tensor = img_tensor.to(DEVICE).requires_grad_(True)

        # Forward pass
        output = self.model(img_tensor)           # [1, n_classes]

        # Backward pass for target class
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        # Grad-CAM formula:
        # 1. Global average pool the gradients → importance weights per channel
        # 2. Weighted sum of activation maps
        # 3. ReLU to keep only positive influence
        grads   = self.gradients[0]               # [C, H, W]
        acts    = self.activations[0]             # [C, H, W]
        weights = grads.mean(dim=(1, 2))          # [C] — global avg pool of grads

        cam = torch.zeros(acts.shape[1:], device=DEVICE)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = F.relu(cam)                         # keep positive influence only

        # Normalise to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy()                  # [30, 3]

# =============================================================================
# STEP 7 — PER-UUID EXPLAINABILITY WITH GRAD-CAM
# =============================================================================
def explain_uuid(model, img_tensor, raw_matrix, uid, pred_label,
                 confidence, proba, le, grad_cam, row_i):
    """
    Generate Grad-CAM explanation for one UUID.
    Saves a 3-panel figure:
      Panel 1: Original normalised image (the input the model saw)
      Panel 2: Grad-CAM heatmap (where the model looked)
      Panel 3: Overlay (heatmap on top of original image)
    """
    class_idx = int(le.transform([pred_label])[0])
    heatmap   = grad_cam.compute(img_tensor, class_idx)   # [30, 3]

    # ── Build figure ────────────────────────────────────────────────────
    normed_img = img_tensor.squeeze().detach().cpu().numpy()  # [30, 3]

    fig, axes = plt.subplots(1, 3, figsize=(14, 7))
    fig.suptitle(
        f"UUID: {uid[:16]}...   →   Predicted: {pred_label}   "
        f"({confidence*100:.1f}% confidence)",
        fontsize=12, fontweight="bold", y=1.01
    )

    # Panel 1 — Original image
    ax = axes[0]
    im = ax.imshow(normed_img, cmap="viridis", aspect="auto",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("Input Image\n(UUID 30-day matrix)", fontsize=10)
    ax.set_xlabel("Features")
    ax.set_ylabel("Days (1 → 30)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["comment\ncount", "avg\nlength", "avg\nsimilarity"], fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2 — Grad-CAM heatmap
    ax = axes[1]
    im2 = ax.imshow(heatmap, cmap="jet", aspect="auto",
                    vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("Grad-CAM Heatmap\n(Red = model focused here)", fontsize=10)
    ax.set_xlabel("Features")
    ax.set_ylabel("Days (1 → 30)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["comment\ncount", "avg\nlength", "avg\nsimilarity"], fontsize=8)
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # Panel 3 — Overlay
    ax = axes[2]
    ax.imshow(normed_img, cmap="gray", aspect="auto",
              vmin=0, vmax=1, interpolation="nearest", alpha=0.6)
    ax.imshow(heatmap, cmap="jet", aspect="auto",
              vmin=0, vmax=1, interpolation="nearest", alpha=0.5)
    ax.set_title("Overlay\n(Heatmap on Input)", fontsize=10)
    ax.set_xlabel("Features")
    ax.set_ylabel("Days (1 → 30)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["comment\ncount", "avg\nlength", "avg\nsimilarity"], fontsize=8)

    # Probability bar below
    prob_text = "   ".join([f"{cls}: {p*100:.1f}%" for cls, p in zip(le.classes_, proba)])
    fig.text(0.5, -0.02, f"Class probabilities: {prob_text}",
             ha="center", fontsize=9, color="dimgray")

    plt.tight_layout()
    fname = f"gradcam_{uid[:8]}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()

    # ── Console explanation ──────────────────────────────────────────────
    # Find which day and feature the model focused on most
    top_day_idx, top_feat_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    top_feat_name = FEATURE_COLS[top_feat_idx]
    top_cam_val   = heatmap[top_day_idx, top_feat_idx]

    # Top 3 hotspots
    flat_idx  = np.argsort(heatmap.ravel())[::-1][:5]
    hotspots  = [(i // N_FEATURES + 1, FEATURE_COLS[i % N_FEATURES], heatmap.ravel()[i])
                 for i in flat_idx]

    print(f"\n  {'─'*56}")
    print(f"  UUID       : {uid}")
    print(f"  Prediction : {pred_label}  ({confidence*100:.1f}% confidence)")
    print(f"\n  Probability breakdown:")
    for cls, prob in zip(le.classes_, proba):
        bar = "█" * int(prob * 25)
        print(f"    {cls:<12} {prob*100:5.1f}%  {bar}")

    print(f"\n  Grad-CAM — Where the model looked (top 5 hotspots):")
    print(f"  {'Day':<8} {'Feature':<28} {'Attention':<10}")
    print(f"  {'─'*46}")
    for day, feat, score in hotspots:
        bar = "█" * int(score * 20)
        print(f"  Day {day:<4} {feat:<28} {score:.3f}  {bar}")

    print(f"\n  Primary focus: Day {top_day_idx+1}, '{top_feat_name}' "
          f"(attention={top_cam_val:.3f})")

    # Natural language explanation
    print(f"\n  Explanation:")
    print(f"  The model classified UUID {uid[:8]}... as [{pred_label}]")
    print(f"  primarily because it focused on '{top_feat_name}' around Day {top_day_idx+1}.")

    # Check what that day's raw values look like
    raw_day = raw_matrix[top_day_idx]
    for feat_name, val in zip(FEATURE_COLS, raw_day):
        print(f"    {feat_name} on Day {top_day_idx+1} = {val:.2f}")

    print(f"  Saved: {fname}")
    return heatmap, fname

# =============================================================================
# STEP 8 — SAVE AND LOAD MODEL
# =============================================================================
def save_model(model, le, feature_stats, path=MODEL_SAVE_PATH):
    torch.save({
        "model_state"  : model.state_dict(),
        "label_encoder": le,
        "feature_stats": feature_stats,
        "n_classes"    : len(le.classes_),
    }, path)
    print(f"\n  Model saved → {path}")

def load_model_bundle(path=MODEL_SAVE_PATH):
    bundle = torch.load(path, map_location=DEVICE, weights_only=False)
    model  = ReconVisionCNN(bundle["n_classes"]).to(DEVICE)
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return model, bundle["label_encoder"], bundle["feature_stats"]

# =============================================================================
# STEP 9 — INFERENCE WITH PER-UUID GRAD-CAM
# =============================================================================
def run_inference(new_csv_path, model_path=MODEL_SAVE_PATH, explain=True):
    """
    Score new UUIDs with per-UUID Grad-CAM explanation.
    new_csv_path : raw CSV, same format as training (label column optional)
    explain      : if True, generate Grad-CAM image per UUID
    """
    print(f"\n{'='*60}\nINFERENCE — Vision CNN Scoring\n{'='*60}")

    model, le, feature_stats = load_model_bundle(model_path)
    raw_df = load_data(new_csv_path)
    if LABEL_COL not in raw_df.columns:
        raw_df[LABEL_COL] = "UNKNOWN"

    images, _, uuid_ids, matrices = build_image_dataset(
        raw_df, feature_stats, label_encoder=None)

    # Score all UUIDs
    dataset = ReconImageDataset(images)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE)

    model.eval()
    all_probs, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            imgs   = batch.to(DEVICE)
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            all_probs.extend(probs)
            all_preds.extend(preds)

    pred_labels = le.inverse_transform(all_preds)
    all_probs   = np.array(all_probs)

    # Build results table
    results = pd.DataFrame({"uuid": uuid_ids, "predicted_class": pred_labels})
    for i, cls in enumerate(le.classes_):
        results[f"prob_{cls}"] = all_probs[:, i].round(4)
    results["confidence"] = all_probs.max(axis=1).round(4)

    print(f"\n  Scored {len(results)} UUIDs:")
    print(results.to_string(index=False))
    results.to_csv("cnn_inference_results.csv", index=False)
    print("\n  Saved: cnn_inference_results.csv")

    # Per-UUID Grad-CAM
    if explain:
        print(f"\n{'='*60}\nPER-UUID GRAD-CAM EXPLAINABILITY\n{'='*60}")
        grad_cam = GradCAM(model, target_layer=model.conv3)

        explanation_rows = []
        for i, (uid, pred_label) in enumerate(zip(uuid_ids, pred_labels)):
            img_tensor  = torch.tensor(images[i:i+1], dtype=torch.float32)
            proba       = all_probs[i]
            confidence  = proba.max()
            heatmap, fname = explain_uuid(
                model, img_tensor, matrices[i], uid,
                pred_label, confidence, proba, le, grad_cam, i
            )

            # Top hotspot for summary
            top_day_idx, top_feat_idx = np.unravel_index(
                np.argmax(heatmap), heatmap.shape)
            explanation_rows.append({
                "uuid"            : uid,
                "predicted_class" : pred_label,
                "confidence"      : round(float(confidence), 4),
                "primary_focus_day"  : top_day_idx + 1,
                "primary_focus_feature": FEATURE_COLS[top_feat_idx],
                "attention_score" : round(float(heatmap.max()), 4),
                "chart_file"      : fname,
            })

        exp_df = pd.DataFrame(explanation_rows)
        exp_df.to_csv("cnn_explanation_summary.csv", index=False)
        print(f"\n  {'='*56}")
        print(f"  Saved: cnn_explanation_summary.csv")
        print(f"  Saved: gradcam_<uuid>.png per UUID")

    return results

# =============================================================================
# MAIN — Full pipeline
# =============================================================================
def main():
    print("\n" + "=" * 60)
    print("  OPTUM CLAIM RECON — VISION CNN PIPELINE")
    print("=" * 60)

    # Step 1: Load
    raw_df = load_data(CSV_PATH)

    # Step 2: Label encode
    le = LabelEncoder()
    le.fit(raw_df[LABEL_COL].unique())
    print(f"\n  Classes: {list(le.classes_)}")

    # Step 3: Compute feature stats for normalisation (from training data)
    feature_stats = compute_feature_stats(raw_df)
    print(f"\n  Feature stats (min, max):")
    for feat, (fmin, fmax) in feature_stats.items():
        print(f"    {feat}: ({fmin:.2f}, {fmax:.2f})")

    # Step 4: Build image matrices
    images, labels, uuid_ids, matrices = build_image_dataset(
        raw_df, feature_stats, label_encoder=le)

    # Step 5: Train
    model = train_model(images, labels, le, n_classes=len(le.classes_))

    # Step 6: Evaluate
    evaluate_model(model, images, labels, le)

    # Step 7: Per-UUID Grad-CAM on training data (first 4 as examples)
    print(f"\n{'='*60}\nSTEP 7 — GRAD-CAM EXPLAINABILITY (sample UUIDs)\n{'='*60}")
    grad_cam = GradCAM(model, target_layer=model.conv3)
    n_explain = min(4, len(uuid_ids))   # explain first 4 during training

    for i in range(n_explain):
        img_tensor = torch.tensor(images[i:i+1], dtype=torch.float32)
        with torch.no_grad():
            logits = model(img_tensor.to(DEVICE))
            proba  = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_enc   = int(proba.argmax())
        pred_label = le.inverse_transform([pred_enc])[0]
        confidence = proba.max()

        explain_uuid(model, img_tensor, matrices[i], uuid_ids[i],
                     pred_label, confidence, proba, le, grad_cam, i)

    # Step 8: Save model
    save_model(model, le, feature_stats)

    # Step 9: Inference example (uncomment when you have new data)
    # run_inference("claudetask_inference.csv")

    print(f"\n{'='*60}\n  PIPELINE COMPLETE\n{'='*60}")
    print("  cnn_confusion_matrix.png     — evaluation")
    print("  gradcam_<uuid>.png           — per-UUID Grad-CAM during training")
    print("  recon_vision_cnn.pt          — saved model")
    print("  cnn_inference_results.csv    — inference scores (after inference)")
    print("  cnn_explanation_summary.csv  — Grad-CAM summary (after inference)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
