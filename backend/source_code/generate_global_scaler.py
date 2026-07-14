import os
import glob
import cv2
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Import CFG and feature extraction from our utilities
# Import CFG and feature extraction from our utilities
from source_code.clients_utils import (
    extract_morphological_features,
    CFG,
    FEATURE_COLS,
    get_partition_config,
)


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    all_features = []
    all_weights = []

    print("Scanning data directory to calculate global morphological features...")

    # Iterate over all partition folders
    partition_folders = sorted(glob.glob(os.path.join(data_dir, "partition_*")))

    if not partition_folders:
        print(f"Error: No partition folders found in {data_dir}/")
        return

    for part_dir in partition_folders:
        part_config = get_partition_config(part_dir)
        cm_per_pixel = part_config.get("cm_per_pixel", CFG["cm_per_pixel"])

        # We only use training data to fit the scaler to prevent data leakage
        train_img_dir = os.path.join(part_dir, "train", "images")
        train_mask_dir = os.path.join(part_dir, "train", "masks")
        weights_csv = os.path.join(part_dir, "actual_weights.csv")

        if not os.path.exists(weights_csv):
            print(f"Skipping {part_dir}: actual_weights.csv not found.")
            continue

        weight_df = pd.read_csv(weights_csv)
        # Ensure column name is standardized
        weight_df["image_name"] = weight_df["image_name"].str.rsplit(".", n=1).str[0]

        # Determine the target column (anything not named 'image_name')
        name_like = {"image_name", "image", "filename", "file", "name"}
        weight_col = [c for c in weight_df.columns if c.lower() not in name_like][0]
        weight_df = weight_df.rename(columns={weight_col: "weight_g"})

        all_weights.append(weight_df[["image_name", "weight_g"]])

        # Process each ground-truth mask to extract perfect features
        mask_files = glob.glob(os.path.join(train_mask_dir, "*.*"))
        for mask_path in mask_files:
            # Read mask in grayscale, normalize to 0/1
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                continue
            mask_2d = (mask_img > 127).astype(np.float32)

            image_id = os.path.splitext(os.path.basename(mask_path))[0]

            # Extract features using our predefined utility
            feat = extract_morphological_features(mask_2d, image_id, cm_per_pixel)
            all_features.append(feat)

    if not all_features:
        print("Error: No features could be extracted. Check your data folders.")
        return

    # Combine all features and weights
    feat_df = pd.DataFrame(all_features)
    weight_df_combined = pd.concat(all_weights, ignore_index=True)

    # Merge on image_name
    merged = pd.merge(feat_df, weight_df_combined, on="image_name", how="inner")

    if merged.empty:
        print(
            "Error: Could not merge extracted features with weights. Check if image names match."
        )
        return

    print(f"Successfully extracted {len(merged)} global samples.")

    # Extract the raw feature columns
    X_raw = merged[FEATURE_COLS].values.astype(np.float32)

    # Fit the global scaler
    print("Fitting StandardScaler on global training data...")
    scaler = StandardScaler()
    scaler.fit(X_raw)

    # Save the scaler
    out_dir = os.path.join(base_dir, "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "global_scaler.pkl")
    joblib.dump(scaler, out_path)

    print(f"\nSuccess! Global scaler permanently saved to: {os.path.abspath(out_path)}")
    print("The FL clients will now use this scaler for all BPNN operations.")


if __name__ == "__main__":
    main()
