"""
=============================================================================
Optum Claim Recon — Bidirectional LSTM Pipeline
=============================================================================
Model     : Bidirectional LSTM with Attention
Classes   : BOT | Non-BOT | Suspect | Prospect
Input     : Same CSV format as XGBoost and Vision CNN pipelines
Explain   : Dual attention weights (forward + backward) per UUID
            + Integrated Gradients heatmap [30 days × 3 features]
Install   : pip install torch pandas scikit-learn matplotlib seaborn
=============================================================================

WHY BIDIRECTIONAL LSTM?
------------------------
Your 30-day sequence has two types of temporal signal:

Forward pass (Day 1 → Day 30):
  "What built up to this point?"
  Catches: ramp-up patterns, gradual escalation of bot behaviour,
           increasing similarity over time

Backward pass (Day 30 → Day 1):
  "What happened after this point?"
  Catches: wind-down patterns, abrupt stops, trailing inactivity
           that only makes sense knowing what came later

Combined: Each day is understood in context of BOTH its history
          and its future — most complete temporal understanding.

EXPLAINABILITY:
  1. Dual Attention Weights  — which days mattered (forward vs backward)
  2. Integrated Gradients    — which day × feature cells drove prediction
     Produces same [30×3] heatmap format as Vision CNN Grad-CAM
     Directly comparable across both models
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib

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
MODEL_SAVE_PATH = "recon_bilstm.pt"
LABEL_COL       = "30lb_bot_flag"
UUID_COL        = "uuid"
DATE_COL        = "ticketsubmissiondate"
WINDOW_DAYS     = 30
FEATURE_COLS    = ["comment_count", "average_comment_length", "average_similarity"]
N_FEATURES      = len(FEATURE_COLS)

# Model hyperparameters
HIDDEN_SIZE     = 64      # hidden units per LSTM direction
N_LAYERS        = 2       # stacked LSTM layers
DROPOUT         = 0.3     # dropout between LSTM layers
N_CLASSES       = 4
BATCH_SIZE      = 16
EPOCHS          = 80
LEARNING_RATE   = 1e-3
RANDOM_SEED     = 42

# Integrated Gradients steps (higher = more accurate, slower)
IG_STEPS        = 50

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

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
    for col in FEATURE_COLS:
        df[col] = df[col].fillna(0)

    df = df.sort_values([UUID_COL, DATE_COL]).reset_index(drop=True)

    if LABEL_COL in df.columns:
        print(f"  Label distribution:\n{df[LABEL_COL].value_counts().to_string()}")
    else:
        print("  Label column absent — inference mode")
    return df

# =============================================================================
# STEP 2 — BUILD SEQUENCE MATRICES
# =============================================================================
def compute_feature_stats(df):
    """Compute per-feature min/max from training data."""
    stats = {}
    for feat in FEATURE_COLS:
        stats[feat] = (float(df[feat].min()), float(df[feat].max()))
    return stats

def normalise_sequence(matrix, feature_stats):
    """Normalise each feature column to [0,1] using training stats."""
    normed = matrix.copy()
    for i, feat in enumerate(FEATURE_COLS):
        fmin, fmax = feature_stats[feat]
        denom = fmax - fmin if fmax > fmin else 1.0
        normed[:, i] = (matrix[:, i] - fmin) / denom
    return np.clip(normed, 0, 1)

def uuid_to_sequence(group, feature_stats):
    """
    Convert one UUID's data to a dense [30 × 3] normalised sequence.
    Missing dates are zero-filled — absence is meaningful signal.
    Returns float32 array [WINDOW_DAYS, N_FEATURES].
    """
    group = group.sort_values(DATE_COL).reset_index(drop=True)
    min_date = group[DATE_COL].min()

    dense = pd.DataFrame({"day_offset": range(WINDOW_DAYS)})
    group["day_offset"] = (group[DATE_COL] - min_date).dt.days
    merged = dense.merge(
        group[["day_offset"] + FEATURE_COLS],
        on="day_offset", how="left"
    )
    merged[FEATURE_COLS] = merged[FEATURE_COLS].fillna(0)

    matrix = merged[FEATURE_COLS].values.astype(np.float32)
    normed = normalise_sequence(matrix, feature_stats)
    return normed, matrix   # normed for model, raw for display

def build_sequences(df, feature_stats, label_encoder=None):
    """
    Build sequence tensors from raw DataFrame.
    Returns:
        sequences : np.array [n_uuids, 30, 3]
        labels    : np.array [n_uuids] int encoded (or None)
        uuid_ids  : list of UUID strings
        raw_mats  : list of raw [30,3] matrices for display
    """
    print(f"\n{'='*60}\nSTEP 2 — BUILDING SEQUENCES\n{'='*60}")
    sequences, labels, uuid_ids, raw_mats = [], [], [], []

    has_label = LABEL_COL in df.columns

    for uid, group in df.groupby(UUID_COL):
        normed, raw = uuid_to_sequence(group, feature_stats)
        sequences.append(normed)
        uuid_ids.append(uid)
        raw_mats.append(raw)

        if has_label and label_encoder is not None:
            lbl = group[LABEL_COL].iloc[0]
            labels.append(int(label_encoder.transform([lbl])[0]))

    sequences = np.array(sequences, dtype=np.float32)  # [N, 30, 3]
    labels    = np.array(labels) if labels else None

    print(f"  Built {len(sequences)} sequences, shape: {sequences[0].shape}")
    print(f"  [n_days={WINDOW_DAYS}, n_features={N_FEATURES}]")
    return sequences, labels, uuid_ids, raw_mats

# =============================================================================
# STEP 3 — PYTORCH DATASET
# =============================================================================
class ReconSequenceDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels    = torch.tensor(labels, dtype=torch.long) \
                         if labels is not None else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if self.labels is not None:
            return seq, self.labels[idx]
        return seq

# =============================================================================
# STEP 4 — BIDIRECTIONAL LSTM WITH ATTENTION
# =============================================================================
class AttentionLayer(nn.Module):
    """
    Additive attention over LSTM hidden states.
    Learns a scalar importance weight per time step (day).
    Returns:
        context   : weighted sum of hidden states [batch, hidden]
        attn_fwd  : attention weights from forward LSTM  [batch, seq_len]
        attn_bwd  : attention weights from backward LSTM [batch, seq_len]
    """
    def __init__(self, hidden_size):
        super().__init__()
        # Separate attention scorers for forward and backward directions
        self.attn_fwd = nn.Linear(hidden_size, 1, bias=False)
        self.attn_bwd = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        """
        hidden_states : [batch, seq_len, 2*hidden_size]
                        Last dim = forward ([:hidden]) + backward ([hidden:])
        """
        h_size = hidden_states.size(-1) // 2
        fwd    = hidden_states[:, :, :h_size]       # [batch, seq, hidden]
        bwd    = hidden_states[:, :, h_size:]        # [batch, seq, hidden]

        # Attention scores per day per direction
        score_fwd = self.attn_fwd(fwd).squeeze(-1)  # [batch, seq]
        score_bwd = self.attn_bwd(bwd).squeeze(-1)  # [batch, seq]

        # Softmax → probability distribution over days
        weight_fwd = F.softmax(score_fwd, dim=1)    # [batch, seq]
        weight_bwd = F.softmax(score_bwd, dim=1)    # [batch, seq]

        # Average the two weight vectors → combined day importance
        weight_combined = (weight_fwd + weight_bwd) / 2.0

        # Context vector — weighted sum of all hidden states
        context = torch.bmm(
            weight_combined.unsqueeze(1),            # [batch, 1, seq]
            hidden_states                            # [batch, seq, 2*hidden]
        ).squeeze(1)                                 # [batch, 2*hidden]

        return context, weight_fwd, weight_bwd


class ReconBiLSTM(nn.Module):
    """
    Bidirectional LSTM with dual attention for recon bot detection.

    Architecture:
      Input [batch, 30, 3]
        ↓
      Linear projection [batch, 30, hidden_size]  — maps 3 features to hidden space
        ↓
      BiLSTM [batch, 30, 2*hidden_size]            — forward + backward passes
        ↓
      AttentionLayer → context [batch, 2*hidden_size]  + attn weights
        ↓
      Dropout
        ↓
      Linear → [batch, n_classes]
    """
    def __init__(self, n_features=N_FEATURES, hidden_size=HIDDEN_SIZE,
                 n_layers=N_LAYERS, dropout=DROPOUT, n_classes=N_CLASSES):
        super().__init__()
        self.hidden_size = hidden_size

        # Project input features to hidden space
        self.input_proj = nn.Linear(n_features, hidden_size)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size   = hidden_size,
            hidden_size  = hidden_size,
            num_layers   = n_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if n_layers > 1 else 0.0,
        )

        # Attention over all time steps
        self.attention = AttentionLayer(hidden_size)

        # Classification head
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(2 * hidden_size, n_classes)

    def forward(self, x, return_attention=False):
        """
        x : [batch, seq_len, n_features]
        Returns logits [batch, n_classes]
        Optionally returns attention weights for explainability
        """
        # Project features
        x = F.relu(self.input_proj(x))              # [batch, 30, hidden]

        # BiLSTM — all hidden states
        lstm_out, _ = self.lstm(x)                  # [batch, 30, 2*hidden]

        # Attention
        context, attn_fwd, attn_bwd = self.attention(lstm_out)

        # Classify
        out = self.fc(self.dropout(context))        # [batch, n_classes]

        if return_attention:
            return out, attn_fwd, attn_bwd
        return out

# =============================================================================
# STEP 5 — TRAINING
# =============================================================================
def train_model(sequences, labels, le):
    print(f"\n{'='*60}\nSTEP 5 — TRAINING BIDIRECTIONAL LSTM\n{'='*60}")

    n_classes = len(le.classes_)
    dataset   = ReconSequenceDataset(sequences, labels)
    n_splits  = min(5, len(np.unique(labels)))

    if n_splits < 2:
        print("  Skipping CV — insufficient class diversity")
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        model  = _train_single(loader, n_classes)
    else:
        skf       = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        fold_accs = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
            train_ds  = ReconSequenceDataset(sequences[train_idx], labels[train_idx])
            val_ds    = ReconSequenceDataset(sequences[val_idx],   labels[val_idx])
            train_ldr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_ldr   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

            fold_model = _train_single(train_ldr, n_classes, verbose=False)
            acc        = _evaluate_accuracy(fold_model, val_ldr)
            fold_accs.append(acc)
            print(f"  Fold {fold+1}/{n_splits} — val accuracy: {acc:.4f}")

        print(f"  CV accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        model  = _train_single(loader, n_classes, verbose=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    return model

def _train_single(loader, n_classes, verbose=True):
    model     = ReconBiLSTM(n_classes=n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch in loader:
            seqs, lbls = batch
            seqs, lbls = seqs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            out  = model(seqs)
            loss = criterion(out, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{EPOCHS}  "
                  f"loss={epoch_loss/len(loader):.4f}")
    return model

def _evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            seqs, lbls = batch
            preds = model(seqs.to(DEVICE)).argmax(dim=1).cpu()
            correct += (preds == lbls).sum().item()
            total   += len(lbls)
    model.train()
    return correct / total if total > 0 else 0.0

# =============================================================================
# STEP 6 — EVALUATE
# =============================================================================
def evaluate_model(model, sequences, labels, le):
    print(f"\n{'='*60}\nSTEP 6 — EVALUATION\n{'='*60}")
    dataset = ReconSequenceDataset(sequences, labels)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE)

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            seqs, lbls = batch
            preds = model(seqs.to(DEVICE)).argmax(dim=1).cpu().numpy()
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
    ax.set_title("Confusion Matrix — BiLSTM", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig("bilstm_confusion_matrix.png", dpi=150)
    plt.close()
    print("  Saved: bilstm_confusion_matrix.png")

# =============================================================================
# EXPLAINABILITY — INTEGRATED GRADIENTS
# =============================================================================
def integrated_gradients(model, seq_tensor, target_class,
                         n_steps=IG_STEPS, baseline=None):
    """
    Integrated Gradients — computes feature attribution for each
    [day, feature] cell in the input sequence.

    Concept:
      Measures how much each input cell contributed to the prediction
      by interpolating from a baseline (all zeros = all absent days)
      to the actual input, and integrating gradients along that path.

    Returns:
      attributions : [30, 3] float array
                     Same shape as input — one importance score per cell
                     Positive = pushed toward predicted class
                     Negative = pushed away from predicted class
    """
    model.eval()

    if baseline is None:
        # Baseline = all zeros (represents a completely absent/inactive UUID)
        baseline = torch.zeros_like(seq_tensor)

    # Interpolate from baseline to actual input across n_steps
    alphas       = torch.linspace(0, 1, n_steps).to(DEVICE)
    interpolated = torch.stack([
        baseline + alpha * (seq_tensor - baseline)
        for alpha in alphas
    ])  # [n_steps, 1, 30, 3]
    interpolated = interpolated.squeeze(1)  # [n_steps, 30, 3]
    interpolated.requires_grad_(True)

    # Forward pass for all interpolated inputs
    output = model(interpolated)            # [n_steps, n_classes]
    target = output[:, target_class].sum()  # scalar
    target.backward()

    # Gradients w.r.t. interpolated inputs
    grads = interpolated.grad.detach()      # [n_steps, 30, 3]

    # Riemann approximation of the integral
    avg_grads    = grads.mean(dim=0)        # [30, 3]
    input_diff   = (seq_tensor.squeeze(0) - baseline.squeeze(0)).detach()
    attributions = (avg_grads * input_diff).cpu().numpy()  # [30, 3]

    return attributions

# =============================================================================
# STEP 7 — PER-UUID EXPLAINABILITY
# =============================================================================
def explain_uuid(model, seq_tensor, raw_matrix, normed_matrix,
                 uid, pred_label, confidence, proba, le,
                 attn_fwd, attn_bwd):
    """
    Generate full explanation for one UUID:
    - Panel 1: Normalised input sequence heatmap
    - Panel 2: Forward attention weights (timeline)
    - Panel 3: Backward attention weights (timeline)
    - Panel 4: Combined attention heatmap [30 × 3]
    - Panel 5: Integrated Gradients attribution [30 × 3]

    Saves a 5-panel figure per UUID.
    """
    class_idx    = int(le.transform([pred_label])[0])

    # ── Integrated Gradients ─────────────────────────────────────────────
    attributions = integrated_gradients(model, seq_tensor, class_idx)
    # Normalise to [-1, 1] for display
    abs_max = np.abs(attributions).max()
    if abs_max > 0:
        attr_normed = attributions / abs_max
    else:
        attr_normed = attributions

    # ── Attention weights ─────────────────────────────────────────────────
    attn_fwd_np  = attn_fwd.squeeze(0).detach().cpu().numpy()   # [30]
    attn_bwd_np  = attn_bwd.squeeze(0).detach().cpu().numpy()   # [30]
    attn_combined= (attn_fwd_np + attn_bwd_np) / 2.0            # [30]

    # Expand combined attention to [30 × 3] for heatmap
    attn_2d      = np.tile(attn_combined[:, np.newaxis], (1, N_FEATURES))

    days         = np.arange(1, WINDOW_DAYS + 1)

    # ── Build 5-panel figure ─────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 9))
    fig.suptitle(
        f"UUID: {uid[:20]}...   →   Predicted: {pred_label}   "
        f"({confidence*100:.1f}% confidence)",
        fontsize=13, fontweight="bold", y=1.01
    )

    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.4)

    # -- Panel 1: Input sequence heatmap --
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(normed_matrix, cmap="viridis", aspect="auto",
                     vmin=0, vmax=1, interpolation="nearest")
    ax1.set_title("Input\n(30-day matrix)", fontsize=9, fontweight="bold")
    ax1.set_xlabel("Features", fontsize=8)
    ax1.set_ylabel("Days (1 → 30)", fontsize=8)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["cc", "len", "sim"], fontsize=7)
    plt.colorbar(im1, ax=ax1, fraction=0.06, pad=0.04)

    # -- Panel 2: Forward attention weights --
    ax2 = fig.add_subplot(gs[1])
    ax2.barh(days, attn_fwd_np, color="steelblue", alpha=0.8)
    ax2.invert_yaxis()
    ax2.set_title("Forward Attention\n(Day 1 → 30)", fontsize=9,
                  fontweight="bold", color="steelblue")
    ax2.set_xlabel("Attention weight", fontsize=8)
    ax2.set_ylabel("Day", fontsize=8)
    ax2.axvline(1.0/WINDOW_DAYS, color="red", linestyle="--",
                linewidth=0.8, label="uniform")
    ax2.legend(fontsize=7)
    ax2.grid(axis="x", alpha=0.3)

    # Top 3 forward days
    top3_fwd = np.argsort(attn_fwd_np)[::-1][:3]
    for d in top3_fwd:
        ax2.barh(d+1, attn_fwd_np[d], color="darkblue", alpha=0.9)

    # -- Panel 3: Backward attention weights --
    ax3 = fig.add_subplot(gs[2])
    ax3.barh(days, attn_bwd_np, color="darkorange", alpha=0.8)
    ax3.invert_yaxis()
    ax3.set_title("Backward Attention\n(Day 30 → 1)", fontsize=9,
                  fontweight="bold", color="darkorange")
    ax3.set_xlabel("Attention weight", fontsize=8)
    ax3.set_ylabel("Day", fontsize=8)
    ax3.axvline(1.0/WINDOW_DAYS, color="red", linestyle="--",
                linewidth=0.8, label="uniform")
    ax3.legend(fontsize=7)
    ax3.grid(axis="x", alpha=0.3)

    top3_bwd = np.argsort(attn_bwd_np)[::-1][:3]
    for d in top3_bwd:
        ax3.barh(d+1, attn_bwd_np[d], color="saddlebrown", alpha=0.9)

    # -- Panel 4: Combined attention heatmap [30×3] --
    ax4 = fig.add_subplot(gs[3])
    im4 = ax4.imshow(attn_2d, cmap="Reds", aspect="auto",
                     vmin=0, interpolation="nearest")
    ax4.set_title("Combined Attention\n(day importance)", fontsize=9,
                  fontweight="bold", color="darkred")
    ax4.set_xlabel("Features", fontsize=8)
    ax4.set_ylabel("Days (1 → 30)", fontsize=8)
    ax4.set_xticks([0, 1, 2])
    ax4.set_xticklabels(["cc", "len", "sim"], fontsize=7)
    plt.colorbar(im4, ax=ax4, fraction=0.06, pad=0.04)

    # -- Panel 5: Integrated Gradients [30×3] --
    ax5 = fig.add_subplot(gs[4])
    im5 = ax5.imshow(attr_normed, cmap="RdBu_r", aspect="auto",
                     vmin=-1, vmax=1, interpolation="nearest")
    ax5.set_title("Integrated Gradients\n(feature attribution)", fontsize=9,
                  fontweight="bold")
    ax5.set_xlabel("Features", fontsize=8)
    ax5.set_ylabel("Days (1 → 30)", fontsize=8)
    ax5.set_xticks([0, 1, 2])
    ax5.set_xticklabels(["cc", "len", "sim"], fontsize=7)
    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.06, pad=0.04)
    cbar5.set_label("← away   toward →", fontsize=7)

    # Probability bar at bottom
    prob_text = "   ".join(
        [f"{cls}: {p*100:.1f}%" for cls, p in zip(le.classes_, proba)])
    fig.text(0.5, -0.02, f"Probabilities: {prob_text}",
             ha="center", fontsize=9, color="dimgray")

    plt.tight_layout()
    fname = f"bilstm_explanation_{uid[:8]}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()

    # ── Console output ────────────────────────────────────────────────────
    top3_fwd_days = np.argsort(attn_fwd_np)[::-1][:3] + 1
    top3_bwd_days = np.argsort(attn_bwd_np)[::-1][:3] + 1

    # Top IG cell
    ig_flat       = np.argsort(np.abs(attributions).ravel())[::-1][:5]
    ig_hotspots   = [(i // N_FEATURES + 1,
                      FEATURE_COLS[i % N_FEATURES],
                      attributions.ravel()[i])
                     for i in ig_flat]

    print(f"\n  {'─'*56}")
    print(f"  UUID         : {uid}")
    print(f"  Prediction   : {pred_label}  ({confidence*100:.1f}% confidence)")

    print(f"\n  Probability breakdown:")
    for cls, prob in zip(le.classes_, proba):
        bar = "█" * int(prob * 25)
        print(f"    {cls:<12} {prob*100:5.1f}%  {bar}")

    print(f"\n  Forward Attention  — top days the model relied on"
          f" (reading Day 1→30):")
    for d in np.argsort(attn_fwd_np)[::-1][:5]:
        bar = "█" * int(attn_fwd_np[d] * 200)
        print(f"    Day {d+1:<3}  weight={attn_fwd_np[d]:.4f}  {bar}")

    print(f"\n  Backward Attention — top days the model relied on"
          f" (reading Day 30→1):")
    for d in np.argsort(attn_bwd_np)[::-1][:5]:
        bar = "█" * int(attn_bwd_np[d] * 200)
        print(f"    Day {d+1:<3}  weight={attn_bwd_np[d]:.4f}  {bar}")

    print(f"\n  Integrated Gradients — top [day × feature] drivers:")
    print(f"  {'Day':<8} {'Feature':<28} {'Attribution':<12} Direction")
    print(f"  {'─'*56}")
    for day, feat, attr in ig_hotspots:
        direction = "→ toward" if attr > 0 else "← against"
        print(f"  Day {day:<4}  {feat:<28} {attr:>+.4f}     {direction} {pred_label}")

    print(f"\n  Natural Language Explanation:")
    print(f"  UUID {uid[:8]}... classified as [{pred_label}] "
          f"({confidence*100:.1f}% confidence)")
    print(f"  → Forward pass focused on days: "
          f"{', '.join([f'Day {d}' for d in top3_fwd_days])}")
    print(f"  → Backward pass focused on days: "
          f"{', '.join([f'Day {d}' for d in top3_bwd_days])}")
    top_ig = ig_hotspots[0]
    direction_nl = "toward" if top_ig[2] > 0 else "against"
    print(f"  → Strongest feature driver: '{top_ig[1]}' on Day {top_ig[0]}"
          f" pushed {direction_nl} {pred_label} "
          f"(attribution={top_ig[2]:+.4f})")

    print(f"  Saved: {fname}")
    return attributions, attn_fwd_np, attn_bwd_np, fname

# =============================================================================
# STEP 8 — SAVE / LOAD MODEL
# =============================================================================
def save_model(model, le, feature_stats, path=MODEL_SAVE_PATH):
    torch.save({
        "model_state"  : model.state_dict(),
        "label_encoder": le,
        "feature_stats": feature_stats,
        "n_classes"    : len(le.classes_),
        "hidden_size"  : HIDDEN_SIZE,
        "n_layers"     : N_LAYERS,
        "dropout"      : DROPOUT,
    }, path)
    print(f"\n  Model saved → {path}")

def load_model_bundle(path=MODEL_SAVE_PATH):
    bundle = torch.load(path, map_location=DEVICE, weights_only=False)
    model  = ReconBiLSTM(
        n_classes   = bundle["n_classes"],
        hidden_size = bundle["hidden_size"],
        n_layers    = bundle["n_layers"],
        dropout     = bundle["dropout"],
    ).to(DEVICE)
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return model, bundle["label_encoder"], bundle["feature_stats"]

# =============================================================================
# STEP 9 — INFERENCE WITH PER-UUID EXPLANATION
# =============================================================================
def run_inference(new_csv_path, model_path=MODEL_SAVE_PATH, explain=True):
    """
    Score new UUIDs with per-UUID dual attention + Integrated Gradients.
    new_csv_path : raw CSV same format as training (label column optional)
    explain      : if True, generate 5-panel explanation per UUID
    """
    print(f"\n{'='*60}\nINFERENCE — BiLSTM Scoring\n{'='*60}")

    model, le, feature_stats = load_model_bundle(model_path)
    raw_df = load_data(new_csv_path)
    if LABEL_COL not in raw_df.columns:
        raw_df[LABEL_COL] = "UNKNOWN"

    sequences, _, uuid_ids, raw_mats = build_sequences(
        raw_df, feature_stats, label_encoder=None)

    # Score all UUIDs
    dataset = ReconSequenceDataset(sequences)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE)

    model.eval()
    all_probs, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            seqs   = batch.to(DEVICE)
            logits = model(seqs)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(probs.argmax(axis=1))

    pred_labels = le.inverse_transform(all_preds)
    all_probs   = np.array(all_probs)

    # Results table
    results = pd.DataFrame({"uuid": uuid_ids, "predicted_class": pred_labels})
    for i, cls in enumerate(le.classes_):
        results[f"prob_{cls}"] = all_probs[:, i].round(4)
    results["confidence"] = all_probs.max(axis=1).round(4)

    print(f"\n  Scored {len(results)} UUIDs:")
    print(results.to_string(index=False))
    results.to_csv("bilstm_inference_results.csv", index=False)
    print("\n  Saved: bilstm_inference_results.csv")

    # Per-UUID explanation
    if explain:
        print(f"\n{'='*60}\nPER-UUID EXPLANATION (Attention + Integrated Gradients)\n{'='*60}")

        explanation_rows = []
        for i, (uid, pred_label) in enumerate(zip(uuid_ids, pred_labels)):
            seq_tensor = torch.tensor(
                sequences[i:i+1], dtype=torch.float32).to(DEVICE)
            proba      = all_probs[i]
            confidence = float(proba.max())

            # Get attention weights
            model.eval()
            with torch.no_grad():
                _, attn_fwd, attn_bwd = model(
                    seq_tensor, return_attention=True)

            # Get normed matrix for display
            normed_mat = sequences[i]   # [30, 3]

            # Full explanation
            attributions, attn_fwd_np, attn_bwd_np, fname = explain_uuid(
                model, seq_tensor, raw_mats[i], normed_mat,
                uid, pred_label, confidence, proba, le,
                attn_fwd, attn_bwd
            )

            # Top drivers for summary
            ig_flat   = np.argsort(np.abs(attributions).ravel())[::-1]
            top_day   = ig_flat[0] // N_FEATURES + 1
            top_feat  = FEATURE_COLS[ig_flat[0] % N_FEATURES]
            top_attr  = float(attributions.ravel()[ig_flat[0]])
            top_fwd_day = int(np.argmax(attn_fwd_np)) + 1
            top_bwd_day = int(np.argmax(attn_bwd_np)) + 1

            explanation_rows.append({
                "uuid"                   : uid,
                "predicted_class"        : pred_label,
                "confidence"             : round(confidence, 4),
                "top_fwd_attention_day"  : top_fwd_day,
                "top_bwd_attention_day"  : top_bwd_day,
                "top_ig_day"             : top_day,
                "top_ig_feature"         : top_feat,
                "top_ig_attribution"     : round(top_attr, 4),
                "chart_file"             : fname,
            })

        exp_df = pd.DataFrame(explanation_rows)
        exp_df.to_csv("bilstm_explanation_summary.csv", index=False)
        print(f"\n  {'='*56}")
        print(f"  Saved: bilstm_inference_results.csv")
        print(f"  Saved: bilstm_explanation_summary.csv")
        print(f"  Saved: bilstm_explanation_<uuid>.png per UUID")

    return results

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 60)
    print("  OPTUM CLAIM RECON — BIDIRECTIONAL LSTM PIPELINE")
    print("=" * 60)

    # Step 1: Load
    raw_df = load_data(CSV_PATH)

    # Step 2: Label encode
    le = LabelEncoder()
    le.fit(raw_df[LABEL_COL].unique())
    print(f"\n  Classes: {list(le.classes_)}")

    # Step 3: Feature stats for normalisation
    feature_stats = compute_feature_stats(raw_df)
    print(f"\n  Feature stats (min, max):")
    for feat, (fmin, fmax) in feature_stats.items():
        print(f"    {feat}: ({fmin:.2f}, {fmax:.2f})")

    # Step 4: Build sequences
    sequences, labels, uuid_ids, raw_mats = build_sequences(
        raw_df, feature_stats, label_encoder=le)

    # Step 5: Train
    model = train_model(sequences, labels, le)

    # Step 6: Evaluate
    evaluate_model(model, sequences, labels, le)

    # Step 7: Explain sample UUIDs during training
    print(f"\n{'='*60}\nSTEP 7 — EXPLANATION (sample UUIDs)\n{'='*60}")
    n_explain = min(4, len(uuid_ids))

    model.eval()
    for i in range(n_explain):
        seq_tensor = torch.tensor(
            sequences[i:i+1], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            logits, attn_fwd, attn_bwd = model(
                seq_tensor, return_attention=True)
            proba = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_enc   = int(proba.argmax())
        pred_label = le.inverse_transform([pred_enc])[0]
        confidence = float(proba.max())

        explain_uuid(
            model, seq_tensor, raw_mats[i], sequences[i],
            uuid_ids[i], pred_label, confidence, proba, le,
            attn_fwd, attn_bwd
        )

    # Step 8: Save model
    save_model(model, le, feature_stats)

    # Step 9: Inference — uncomment when you have new data
    run_inference("claudetask_inference.csv")

    print(f"\n{'='*60}\n  PIPELINE COMPLETE\n{'='*60}")
    print("  bilstm_confusion_matrix.png        — evaluation")
    print("  bilstm_explanation_<uuid>.png      — 5-panel explanation per UUID")
    print("  recon_bilstm.pt                    — saved model")
    print("  bilstm_inference_results.csv       — scores (after inference)")
    print("  bilstm_explanation_summary.csv     — summary (after inference)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

# !pip install torch pandas scikit-learn matplotlib seaborn