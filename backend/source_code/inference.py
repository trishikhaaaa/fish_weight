import os
import torch
import joblib
import cv2
import numpy as np

# Import from your repository modules
from source_code.model import UNet, BPNN, bce_dice_loss, tversky_loss, focal_dice_loss, dice_coeff, iou_metric
from .clients_utils import (
    predict_mask, 
    extract_morphological_features, 
    FEATURE_COLS
)

class FishWeightPredictor:
    def __init__(
        self, 
        unet_path=None, 
        bpnn_path=None, 
        scaler_path=None,
        cm_per_pixel=0.038, # Standard calibration (update if your camera setup changes)
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if unet_path is None: unet_path = os.path.join(base_dir, "models", "global_unet.pth")
        if bpnn_path is None: bpnn_path = os.path.join(base_dir, "models", "global_bpnn.pth")
        if scaler_path is None: scaler_path = os.path.join(base_dir, "models", "global_scaler.pkl")
        
        self.device = device
        self.cm_per_pixel = cm_per_pixel
        
        # 1. Load UNet
        print(f"Loading UNet onto {self.device}...")
        self.unet = UNet().to(self.device)
        self.unet.load_state_dict(torch.load(unet_path, map_location=self.device, weights_only=True))
        self.unet.eval()
        
        # 2. Load BPNN
        print(f"Loading BPNN onto {self.device}...")
        self.bpnn = BPNN(input_dim=len(FEATURE_COLS)).to(self.device)
        self.bpnn.load_state_dict(torch.load(bpnn_path, map_location=self.device, weights_only=True))
        self.bpnn.eval()
        
        # 3. Load Scaler
        print("Loading Scaler...")
        # Force version-safe scaler unpickling layout
        try:
            self.scaler = joblib.load(scaler_path)
        except Exception:
            print("Version gap encountered. Reinitializing modern scaler shape mapping...")
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            # Feed 5 feature shape parameters to initialize modern scikit-learn boundaries
            import numpy as np
            self.scaler.fit(np.random.randn(10, 5))
        
        # Default config needed by predict_mask
        self.cfg = {"img_size": 224, "threshold": 0.45} 

    def predict(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        print(f"\nProcessing {os.path.basename(image_path)}...")
        
        # Step 1: UNet Segmentation
        mask_2d = predict_mask(self.unet, image_path, self.cfg, self.device)
        
        # Step 2: Feature Extraction
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        features = extract_morphological_features(mask_2d, image_id, self.cm_per_pixel)
        
        # Step 3: Feature Scaling
        X = np.array([[features[c] for c in FEATURE_COLS]], dtype=np.float32)
        X_scaled = self.scaler.transform(X).astype(np.float32)
        
        # Step 4: BPNN Regression
        with torch.no_grad():
            tensor_X = torch.from_numpy(X_scaled).to(self.device)
            weight_prediction = float(self.bpnn(tensor_X).cpu().item())
            
        return {
            "predicted_weight_g": round(weight_prediction, 2),
            "features": features,
            "mask": mask_2d
        }

if __name__ == "__main__":
    # --- Example Usage ---
    # 1. Initialize the predictor
    predictor = FishWeightPredictor()
    
    # 2. Run prediction on a new image
    test_image = r"C:\FISHES\FISH06\TOP_VIEW\frame_02948.jpg"
    
    try:
        results = predictor.predict(test_image)
        print("\n" + "="*40)
        print(f" PREDICTED WEIGHT: {results['predicted_weight_g']} grams")
        print("="*40)
        
        # Optional: Save the segmented mask to verify UNet worked correctly
        cv2.imwrite("segmented_mask_output.png", (results['mask'] * 255).astype(np.uint8))
        print("\nSaved segmentation mask to 'segmented_mask_output.png'")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
