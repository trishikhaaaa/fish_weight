"""fl_dp_sa: Flower Example using Differential Privacy and Secure Aggregation."""

import io
import os
import glob
import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from source_code.model import UNet, BPNN, bce_dice_loss, tversky_loss, focal_dice_loss, dice_coeff, iou_metric

CFG = {
    "cm_per_pixel": 0.1,
    "threshold": 0.45,
    "batch_size": 8,
    "epochs": 3,
    "lr": 1e-4,
    "seed": 42,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_COLS = [
    "length_cm",
    "thickness_cm",
    "perimeter_cm",
    "area_cm2",
    "volume_proxy_cm3",
]


def get_partition_config(partition_dir: str) -> dict:
    config_path = os.path.join(partition_dir, "config.toml")
    if not os.path.exists(config_path):
        return {"cm_per_pixel": CFG["cm_per_pixel"]}
    try:
        import tomllib

        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        import toml

        with open(config_path, "r") as f:
            return toml.load(f)


def load_image(path: str, is_mask: bool = False) -> np.ndarray:
    """Returns float32 numpy: (H,W,3) for images, (H,W,1) for masks."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
    img = cv2.copyMakeBorder(img, 0, 8, 0, 0, cv2.BORDER_CONSTANT, value=0)
    if is_mask:
        img = (img > 127).astype(np.float32)[..., np.newaxis]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class FishSegDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, augment: bool = False):
        self.augment = augment
        img_stems = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(image_dir, "*"))
            if os.path.splitext(p)[1].lower() in VALID_EXTS
        }
        msk_stems = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(mask_dir, "*"))
            if os.path.splitext(p)[1].lower() in VALID_EXTS
        }
        common = sorted(img_stems.keys() & msk_stems.keys())
        self.image_paths = [img_stems[k] for k in common]
        self.mask_paths = [msk_stems[k] for k in common]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_image(self.image_paths[idx], is_mask=False)  # (H,W,3)
        mask = load_image(self.mask_paths[idx], is_mask=True)  # (H,W,1)

        if self.augment:
            if np.random.rand() > 0.5:
                img = img[:, ::-1, :].copy()
                mask = mask[:, ::-1, :].copy()
            if np.random.rand() > 0.5:
                img = img[::-1, :, :].copy()
                mask = mask[::-1, :, :].copy()
            delta = np.random.uniform(-0.2, 0.2)
            img = np.clip(img + delta, 0.0, 1.0)

        # (H,W,C) → (C,H,W)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        mask = torch.from_numpy(mask.transpose(2, 0, 1))
        return img, mask


def train_unet_one_epoch(
    model, loader, optimizer, device, global_model=None, proximal_mu=0.0
):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        
        preds = model(imgs)
        loss = focal_dice_loss(preds, masks)

        if global_model is not None and proximal_mu > 0.0:
            proximal_term = 0.0
            for local_weights, global_weights in zip(
                model.parameters(), global_model.parameters()
            ):
                proximal_term += torch.square(local_weights - global_weights).sum()
            loss += (proximal_mu / 2) * proximal_term

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_unet(model, loader, device):
    model.eval()
    val_loss, dice_scores, iou_scores = 0.0, [], []
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            batch_loss = focal_dice_loss(preds, masks).item()
                
            val_loss += batch_loss
            dice_scores.append(dice_coeff(preds, masks))
            iou_scores.append(iou_metric(preds, masks))

            # Binary segmentation stats for extended metrics
            preds_bin = (preds > CFG.get("threshold", 0.45)).float()
            masks_bin = (masks > 0.5).float()

            total_tp += (preds_bin * masks_bin).sum().item()
            total_fp += (preds_bin * (1 - masks_bin)).sum().item()
            total_fn += ((1 - preds_bin) * masks_bin).sum().item()
            total_tn += ((1 - preds_bin) * (1 - masks_bin)).sum().item()

    n = len(loader)

    # Calculate aggregate metrics
    pixel_acc = (total_tp + total_tn) / (
        total_tp + total_tn + total_fp + total_fn + 1e-7
    )
    precision = total_tp / (total_tp + total_fp + 1e-7)
    recall = total_tp / (total_tp + total_fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    # mIoU calculation (average of foreground IoU and background IoU)
    fg_iou = total_tp / (total_tp + total_fp + total_fn + 1e-7)
    bg_iou = total_tn / (total_tn + total_fp + total_fn + 1e-7)
    miou = (fg_iou + bg_iou) / 2.0

    return {
        "val_loss": val_loss / n,
        "dice": float(np.mean(dice_scores)),
        "iou": float(np.mean(iou_scores)),
        "miou": float(miou),
        "pixel_accuracy": float(pixel_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def train_unet(model, image_dir, mask_dir, cfg=CFG, device=DEVICE, global_model=None, proximal_mu=0.0):
    from sklearn.model_selection import train_test_split

    ds = FishSegDataset(image_dir, mask_dir, augment=False)
    idx = list(range(len(ds)))
    t_idx, v_idx = train_test_split(idx, test_size=0.15, random_state=cfg["seed"])

    train_ds = torch.utils.data.Subset(
        FishSegDataset(image_dir, mask_dir, augment=True), t_idx
    )
    val_ds = torch.utils.data.Subset(
        FishSegDataset(image_dir, mask_dir, augment=False), v_idx
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    unet_lr = cfg.get("unet_lr", cfg.get("lr", 1e-4))
    optimizer = optim.AdamW(model.parameters(), lr=unet_lr, weight_decay=1e-4)

    best_dice, patience_count = 0.0, 0
    best_state_bytes = None  # in-memory checkpoint — no disk write
    for epoch in range(cfg["epochs"]):
        tr_loss = train_unet_one_epoch(model, train_loader, optimizer, device, global_model, proximal_mu)
        metrics_dict = eval_unet(model, val_loader, device)
        vl_loss, vl_dice, vl_iou = (
            metrics_dict["val_loss"],
            metrics_dict["dice"],
            metrics_dict["iou"],
        )
        print(
            f"[{epoch+1:03d}] loss={tr_loss:.4f} | val_loss={vl_loss:.4f} dice={vl_dice:.4f} iou={vl_iou:.4f} miou={metrics_dict['miou']:.4f} px_acc={metrics_dict['pixel_accuracy']:.4f}"
        )
        if vl_dice > best_dice:
            best_dice = vl_dice
            buf = io.BytesIO()
            torch.save(model.state_dict(), buf)
            best_state_bytes = buf.getvalue()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 10:
                print("Early stopping triggered.")
                break

    if best_state_bytes is not None:
        model.load_state_dict(
            torch.load(io.BytesIO(best_state_bytes), map_location=device)
        )
    return model


def prepare_bpnn_data(features_csv: str, weights_csv: str, scaler_path: str, cfg=CFG):
    feat_df = pd.read_csv(features_csv)
    weight_df = pd.read_csv(weights_csv)

    for df in [feat_df, weight_df]:
        df["image_name"] = df["image_name"].str.rsplit(".", n=1).str[0]

    name_like = {"image_name", "image", "filename", "file", "name"}
    weight_col = [c for c in weight_df.columns if c.lower() not in name_like][0]
    weight_df = weight_df.rename(columns={weight_col: "weight_g"})

    merged = pd.merge(
        feat_df, weight_df[["image_name", "weight_g"]], on="image_name", how="inner"
    )
    X_raw = merged[FEATURE_COLS].values.astype(np.float32)
    y_raw = merged["weight_g"].values.astype(np.float32)

    try:
        scaler = joblib.load(scaler_path)
    except Exception:
        # Graceful version-fallback calibration setup
        scaler = StandardScaler()
        scaler.fit(X_raw)
        
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    idx = np.random.permutation(len(X_scaled))
    val_size = int(len(idx) * 0.15)
    t_idx, v_idx = idx[val_size:], idx[:val_size]

    return (
        X_scaled[t_idx],
        y_raw[t_idx],
        X_scaled[v_idx],
        y_raw[v_idx],
        scaler,
        merged,
    )


def train_bpnn(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    cfg=CFG,
    device=DEVICE,
    global_model=None,
    proximal_mu=0.0,
):
    X_tr = torch.from_numpy(X_train).to(device)
    y_tr = torch.from_numpy(y_train).to(device)
    X_v = torch.from_numpy(X_val).to(device)
    y_v = torch.from_numpy(y_val).to(device)

    bpnn_lr = cfg.get("bpnn_lr", cfg.get("lr", 1e-4))
    optimizer = optim.AdamW(model.parameters(), lr=bpnn_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15, min_lr=1e-6
    )
    criterion = nn.HuberLoss()

    best_val, patience_count = float("inf"), 0
    best_state_bytes = None

    history = []

    for epoch in range(cfg["epochs"]):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr)

        if global_model is not None and proximal_mu > 0.0:
            proximal_term = 0.0
            for local_weights, global_weights in zip(
                model.parameters(), global_model.parameters()
            ):
                proximal_term += torch.square(local_weights - global_weights).sum()
            loss += (proximal_mu / 2) * proximal_term

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            buf = io.BytesIO()
            torch.save(model.state_dict(), buf)
            best_state_bytes = buf.getvalue()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 40:
                print(f"Early stop at epoch {epoch+1}")
                break

        epoch_metrics = eval_bpnn(model, X_val, y_val, device, verbose=False)
        epoch_metrics["epoch"] = epoch + 1
        epoch_metrics["train_mse"] = loss.item()
        epoch_metrics["val_mse"] = val_loss
        history.append(epoch_metrics)

        if (epoch + 1) % 20 == 0:
            print(
                f"[{epoch+1:03d}] train_mse={loss.item():.4f} val_mse={val_loss:.4f} | MAE={epoch_metrics['mae']:.2f} RMSE={epoch_metrics['rmse']:.2f} R²={epoch_metrics['r2']:.4f} MAPE={epoch_metrics['mape']:.2f}%"
            )

    if best_state_bytes is not None:
        model.load_state_dict(
            torch.load(io.BytesIO(best_state_bytes), map_location=device)
        )
    return model, best_val, history


def eval_bpnn(model, X_val, y_val, device=DEVICE, verbose=True):
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_val).to(device)).cpu().numpy()
    mse = mean_squared_error(y_val, preds)
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, preds)
    mape = np.mean(np.abs((y_val - preds) / (y_val + 1e-6))) * 100
    if verbose:
        print(
            f"MAE={mae:.2f}g  MSE={mse:.2f}g  RMSE={rmse:.2f}g  R²={r2:.4f}  MAPE={mape:.2f}%"
        )
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}


def extract_morphological_features(
    mask_2d: np.ndarray, image_id: str, cm_per_pixel: float = CFG["cm_per_pixel"]
) -> dict:
    feats = dict(
        image_name=image_id,
        length_cm=0.0,
        thickness_cm=0.0,
        perimeter_cm=0.0,
        area_cm2=0.0,
        volume_proxy_cm3=0.0,
    )
    mask_u8 = (mask_2d * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return feats
    c = max(contours, key=cv2.contourArea)
    if len(c) < 5:
        return feats

    _, (d1, d2), _ = cv2.fitEllipse(c)
    L = max(d1, d2) * cm_per_pixel
    T = min(d1, d2) * cm_per_pixel
    feats.update(
        length_cm=round(L, 4),
        thickness_cm=round(T, 4),
        perimeter_cm=round(cv2.arcLength(c, True) * cm_per_pixel, 4),
        area_cm2=round(cv2.contourArea(c) * cm_per_pixel**2, 4),
        volume_proxy_cm3=round((np.pi / 6) * L * T**2, 4),
    )
    return feats


def predict_mask(model: UNet, image_path: str, cfg=CFG, device=DEVICE) -> np.ndarray:
    """Returns clean isolated float32 (H,W) binary mask matching backend design."""
    img = load_image(image_path, is_mask=False)
    inp = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        prob = model(inp)[0, 0].cpu().numpy()
        
    # High-precision thresholding setup
    bin_mask = (prob > cfg.get("threshold", 0.45)).astype(np.uint8) * 255
    
    # CRITICAL FIX: Eliminate background tank lip noise and edge scattering arrays
    clean_mask = np.zeros_like(bin_mask)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
        
    closed = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return (closed / 255.0).astype(np.float32)


def run_pipeline_single(
    image_path: str,
    unet_model: UNet,
    bpnn_model: BPNN,
    scaler,
    cm_per_pixel: float,
    cfg=CFG,
    device=DEVICE,
) -> dict:
    mask_2d = predict_mask(unet_model, image_path, cfg, device)
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    feat = extract_morphological_features(mask_2d, image_id, cm_per_pixel)

    X = np.array([[feat[c] for c in FEATURE_COLS]], dtype=np.float32)
    
    # Safe verification for scaler transform matching fallback setups
    try:
        X_sc = scaler.transform(X).astype(np.float32)
    except Exception:
        X_sc = X
        
    bpnn_model.eval()
    with torch.no_grad():
        weight_g = float(bpnn_model(torch.from_numpy(X_sc).to(device)).cpu().item())

    feat["predicted_weight_g"] = round(weight_g, 2)
    return feat


def evaluate_on_folder(
    unet_model,
    bpnn_model,
    scaler,
    image_dir,
    labels_csv,
    cfg=CFG,
    device=DEVICE,
    eval_feat_csv=None,
):
    partition_dir = os.path.dirname(os.path.dirname(image_dir))
    part_config = get_partition_config(partition_dir)
    cm_per_pixel = part_config.get("cm_per_pixel", cfg["cm_per_pixel"])

    df_labels = pd.read_csv(labels_csv)
    df_labels["image_name"] = df_labels["image_name"].str.rsplit(".", n=1).str[0]

    name_like = {"image_name", "image", "filename", "file", "name"}
    weight_col = [c for c in df_labels.columns if c.lower() not in name_like][0]
    df_labels = df_labels.rename(columns={weight_col: "weight_g"})

    results = []
    for p in sorted(glob.glob(os.path.join(image_dir, "*"))):
        if os.path.splitext(p)[1].lower() not in VALID_EXTS:
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        row = df_labels[df_labels["image_name"] == stem]
        if row.empty:
            continue
        out = run_pipeline_single(
            p, unet_model, bpnn_model, scaler, cm_per_pixel, cfg, device
        )
        row_dict = {
            "image_name": stem,
            "actual_weight_g": float(row["weight_g"].values[0]),
        }
        row_dict.update(out)
        results.append(row_dict)

    df = pd.DataFrame(results)
    if df.empty:
        print("[evaluate_on_folder] No matched results found.")
        return df

    csv_out_path = (
        eval_feat_csv
        if eval_feat_csv
        else os.path.join(partition_dir, "features_eval.csv")
    )
    df.to_csv(csv_out_path, index=False)
    y_true, y_pred = df["actual_weight_g"].values, df["predicted_weight_g"].values
    print(
        f"MAE={mean_absolute_error(y_true,y_pred):.4f}  "
        f"RMSE={np.sqrt(mean_squared_error(y_true,y_pred)):.4f}  "
        f"R²={r2_score(y_true,y_pred):.4f}  "
        f"MAPE={np.mean(np.abs((y_true-y_pred)/(y_true+1e-7)))*100:.2f}%"
    )
    return df