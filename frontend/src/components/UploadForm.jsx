import "../styles/UploadForm.css";
import { useState } from "react";
import ResultCard from "./ResultCard";
import { useDropzone } from "react-dropzone";

function UploadForm() {
  const [image, setImage] = useState(null);
  const [weight, setWeight] = useState("");
  const [cmPerPixel, setCmPerPixel] = useState("0.038"); // Default alignment matching backend[cite: 3, 11]
  const [result, setResult] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const onDropImage = (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setImage(file);
      setImagePreview(URL.createObjectURL(file));
      setError("");
      setResult(null);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: onDropImage,
    accept: { "image/*": [] },
    multiple: false,
    disabled: loading, // Prevent drops while processing
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      setError("Please upload a fish image.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("image", image);
      formData.append("cm_per_pixel", cmPerPixel);
      if (weight) formData.append("actual_weight", weight);

      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed");
      }

      setResult({
        predicted_weight_g: data.predicted_weight_g,
        length_cm: data.features.length_cm,
        thickness_cm: data.features.thickness_cm,
        perimeter_cm: data.features.perimeter_cm,
        area_cm2: data.features.area_cm2,
        volume_proxy_cm3: data.features.volume_proxy_cm3,
        generated_mask: data.generated_mask || null,
      });
    } catch (err) {
      console.error(err);
      setError(err.message || "An unexpected network error occurred.");
    } finally {
      setLoading(false);
    }
  };

  const removeImage = () => {
    setImage(null);
    setImagePreview(null);
    setResult(null);
    setError("");
  };

  return (
    <div className="dashboard-container">
      <div className={`form-panel ${loading ? "processing" : ""}`}>
        <h2>Analysis Configurations</h2>
        <form onSubmit={handleSubmit}>
          <label>Upload Fish Image</label>
          <div {...getRootProps()} className={`dropzone ${isDragActive ? "active" : ""} ${loading ? "disabled" : ""}`}>
            <input {...getInputProps()} />
            <p>{isDragActive ? "Drop the file here..." : "Drag & Drop Fish Image Here"}</p>
            <span>or click to browse local files</span>
          </div>

          <label>Camera Calibration (cm/pixel)</label>
          <input
            type="number"
            step="0.0001"
            value={cmPerPixel}
            disabled={loading}
            onChange={(e) => setCmPerPixel(e.target.value)}
          />
          <small className="input-hint">Baseline scale dimension matching setup lens height.</small>

          <label>Actual Weight (Optional Reference)</label>
          <input
            type="number"
            value={weight}
            disabled={loading}
            placeholder="Enter baseline weight (grams)"
            onChange={(e) => setWeight(e.target.value)}
          />

          {error && <p className="error-message">{error}</p>}

          <button type="submit" disabled={loading}>
            {loading ? (
              <span className="spinner-container">
                <span className="loading-spinner"></span> Processing AI Models...
              </span>
            ) : (
              "Run Weight Prediction Pipeline"
            )}
          </button>
        </form>
      </div>

      <div className="display-panel">
        {!result && imagePreview && (
          <div className="preview-card-standalone">
            <h3>Uploaded Working Specimen</h3>
            <div className="img-wrapper">
              <img src={imagePreview} alt="Uploaded Fish specimen" />
            </div>
            <button type="button" className="remove-btn" onClick={removeImage} disabled={loading}>
              Clear Workspace Image
            </button>
          </div>
        )}
        <ResultCard result={result} originalImage={imagePreview} />
        {!imagePreview && !result && (
          <div className="empty-workspace-state">
            <p>Upload a localized fish profile specimen photograph to initialize the segmentation models.</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default UploadForm;