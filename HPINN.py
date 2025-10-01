# pinn_stacking.py
"""
Stacking Residual PINN with RF Uncertainty
- Load dataset (Excel/CSV)
- Use an existing RandomForestRegressor (rf_prior_model.pkl) as prior; if missing, train & save one.
- Compute out-of-fold (OOF) RF predictions and per-sample std across trees (uncertainty).
- Train a residual NN that takes [X_scaled, rf_oof, rf_std] as input and outputs (mu_res, logvar).
  Final prediction = rf_oof + mu_res.
- Loss: heteroscedastic NLL = 0.5 * ( (y - yhat)^156 / var + logvar ).
- Early stopping based on validation R².
- Save residual model and scaler, and save validation comparison plot.
"""
import os
import math
import joblib
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "E:\Wangchuang_pythonProject\pytorch_env]\Model_DataAnalysis/Train_Data.xlsx"
TARGET_COL = "Irregularity"
RF_PATH = "rf_prior_model.pkl"
RESIDUAL_OUT = "residual_net_state.pt"
SCALER_OUT = "residual_scaler.npy"
PLOT_OUT = "validation_comparison.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 10

# training hyperparams
N_SPLITS_OOF = 5
TEST_SIZE = 0.2
BATCH_SIZE = 32
LR = 5e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 800
PATIENCE = 80
CLAMP_LOGVAR = (-12.0, 6.0)
RANDOM_STATE = 42

# -----------------------------
# Utilities
# -----------------------------
def load_data(path: str, target_col: str) -> Tuple[np.ndarray, np.ndarray, list]:
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {df.columns.tolist()}")
    features = [c for c in df.columns if c != target_col]
    X = df[features].values.astype(float)
    y = df[target_col].values.astype(float)
    return X, y, features

def ensure_rf_model(X: np.ndarray, y: np.ndarray, rf_path: str) -> RandomForestRegressor:
    if os.path.exists(rf_path):
        try:
            rf = joblib.load(rf_path)
            print(f"Loaded RF from {rf_path}")
            return rf
        except Exception as e:
            print(f"Warning: could not load RF ({e}), will retrain.")
    print("Training RF on full data (fallback)...")
    rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    joblib.dump(rf, rf_path)
    print(f"Saved RF to {rf_path}")
    return rf

def compute_oof_predictions_with_std(X: np.ndarray, y: np.ndarray, rf_model: RandomForestRegressor, n_splits: int = 5):
    """Compute out-of-fold RF mean preds and per-sample std across trees"""
    params = rf_model.get_params()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_mean = np.zeros_like(y, dtype=float)
    oof_std = np.zeros_like(y, dtype=float)
    print(f"Generating {n_splits}-fold OOF preds+uncertainty using RF params: "
          f"{ {k:v for k,v in params.items() if k in ['n_estimators','max_depth','max_features']} }")
    for fold, (tr, te) in enumerate(kf.split(X), start=1):
        rf_cv = RandomForestRegressor(**params)
        rf_cv.fit(X[tr], y[tr])
        preds_all = np.stack([tree.predict(X[te]) for tree in rf_cv.estimators_], axis=0)
        oof_mean[te] = preds_all.mean(axis=0)
        oof_std[te] = preds_all.std(axis=0)
        print(f"  OOF fold {fold}: trained on {len(tr)} -> predicted {len(te)}")
    return oof_mean, oof_std

# -----------------------------
# Residual network
# -----------------------------
class ResidualNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.fe = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, 1)
        self.logvar_head = nn.Linear(prev, 1)

    def forward(self, x):
        h = self.fe(x)
        mu = self.mu_head(h).squeeze(-1)
        logvar = self.logvar_head(h).squeeze(-1)
        return mu, logvar

def heteroscedastic_nll(y_true: torch.Tensor, y_pred: torch.Tensor, logvar: torch.Tensor):
    logvar = torch.clamp(logvar, CLAMP_LOGVAR[0], CLAMP_LOGVAR[1])
    var = torch.exp(logvar)
    se = (y_true - y_pred) ** 2
    nll = 0.5 * (se / var + logvar)
    return nll.mean()

# -----------------------------
# Training function
# -----------------------------
def train_stacking_pinn(
    X: np.ndarray,
    y: np.ndarray,
    rf_model: RandomForestRegressor,
    features: list,
):
    X_proc = X.astype(float).copy()
    X_proc[np.isnan(X_proc)] = 0.0
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_proc)
    np.save(SCALER_OUT, {"mean": scaler.mean_, "scale": scaler.scale_})
    print(f"Saved scaler -> {SCALER_OUT}")

    # OOF mean & std
    y_rf_oof, y_rf_std = compute_oof_predictions_with_std(X_scaled, y, rf_model, n_splits=N_SPLITS_OOF)
    # Full model preds & std (for eval only)
    preds_all_full = np.stack([tree.predict(X_scaled) for tree in rf_model.estimators_], axis=0)
    y_rf_full = preds_all_full.mean(axis=0)
    y_rf_full_std = preds_all_full.std(axis=0)

    # Stacking input: add both rf_oof and rf_std
    X_stack = np.concatenate([X_scaled, y_rf_oof.reshape(-1, 1), y_rf_std.reshape(-1, 1)], axis=1)

    # train/val split
    idxs = np.arange(len(y))
    train_idx, val_idx = train_test_split(idxs, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_tr, y_tr = X_stack[train_idx], y[train_idx]
    X_val, y_val = X_stack[val_idx], y[val_idx]
    rf_tr, rf_val = y_rf_oof[train_idx], y_rf_oof[val_idx]

    print(f"Train/Val sizes: {len(train_idx)}/{len(val_idx)}")

    device = DEVICE
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    rf_tr_t = torch.tensor(rf_tr, dtype=torch.float32, device=device)

    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    rf_val_t = torch.tensor(rf_val, dtype=torch.float32, device=device)

    input_dim = X_stack.shape[1]
    model = ResidualNet(input_dim=input_dim, hidden_dims=(128, 64), dropout=0.2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    n_train = X_tr.shape[0]
    best_val_r2 = -np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = np.random.permutation(n_train)
        losses = []
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]
            rf_b = rf_tr_t[idx]
            optimizer.zero_grad()
            mu_res, logvar = model(xb)
            mu_res = mu_res.view(-1)
            logvar = logvar.view(-1)
            yhat = rf_b + mu_res
            loss = heteroscedastic_nll(yb, yhat, logvar)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            mu_val, logvar_val = model(X_val_t)
            mu_val = mu_val.view(-1)
            logvar_val = logvar_val.view(-1)
            yhat_val = rf_val_t + mu_val
            val_pred_np = yhat_val.cpu().numpy().reshape(-1)
            val_r2 = r2_score(y_val, val_pred_np)
            val_rmse = math.sqrt(mean_squared_error(y_val, val_pred_np))
            rf_val_r2 = r2_score(y_val, rf_val)
            rf_val_rmse = math.sqrt(mean_squared_error(y_val, rf_val))
            val_nll = heteroscedastic_nll(y_val_t, yhat_val, logvar_val).item()

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}: TrainLoss={np.mean(losses):.6e}, ValNLL={val_nll:.6e}, "
                  f"Val R²={val_r2:.4f}, RF Val R²={rf_val_r2:.4f}")

        if val_r2 > best_val_r2 + 1e-9:
            best_val_r2 = val_r2
            best_state = {
                "model_state": model.state_dict(),
                "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
                "rf_params": rf_model.get_params()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}, best Val R²={best_val_r2:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
    torch.save({"model_state": model.state_dict(),
                "scaler": best_state["scaler"] if best_state else {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}},
               RESIDUAL_OUT)
    print(f"Saved residual model -> {RESIDUAL_OUT}")

    model.eval()
    with torch.no_grad():
        mu_val, _ = model(X_val_t)
        mu_val = mu_val.view(-1).cpu().numpy()
        rf_val_np = rf_val
        pred_val = rf_val_np + mu_val

        # ===== Final Evaluation with extra metrics =====
        def evaluate_metrics(name, y_true, y_pred):
            r2 = r2_score(y_true, y_pred)
            rmse = math.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            epsilon = 1e-8  # 避免除0
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
            print(f"{name:12s}: R²={r2:.4f}, RMSE={rmse:.6f}, MAE={mae:.6f}, MAPE={mape:.2f}%")

        print("\n===== Final Evaluation (validation set) =====")
        evaluate_metrics("RF", y_val, rf_val_np)
        evaluate_metrics("ResidualPINN", y_val, pred_val)

    plot_validation(y_val, rf_val, pred_val, PLOT_OUT)
    print(f"Saved validation comparison plot -> {PLOT_OUT}")
    return model, scaler, (y_val, rf_val, pred_val)

# -----------------------------
# Plot helper
# -----------------------------
def plot_validation(y_true, rf_pred, res_pred, out_fname):
    order = np.argsort(y_true)
    y_true_s = y_true[order]
    rf_s = rf_pred[order]
    res_s = res_pred[order]
    plt.figure(figsize=(10,5))
    plt.plot(y_true_s, label="Ground Truth", marker='o')
    plt.plot(rf_s, label="RF (OOF)", marker='x')
    plt.plot(res_s, label="ResidualPINN", marker='s')
    plt.xlabel("validation samples (sorted by GT)")
    plt.ylabel("Target")
    plt.legend()
    plt.title("Validation: RF vs ResidualPINN vs GT")
    plt.tight_layout()
    plt.savefig(out_fname)
    plt.close()

# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading data:", DATA_PATH)
    X, y, features = load_data(DATA_PATH, TARGET_COL)
    print(f"Data shape: {X.shape[0]} samples, {X.shape[1]} features")
    print("Target:", TARGET_COL)
    rf = ensure_rf_model(X, y, RF_PATH)
    train_stacking_pinn(X, y, rf, features)
    print("Done.")

if __name__ == "__main__":
    main()
