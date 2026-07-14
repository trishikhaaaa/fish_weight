import "../styles/ResultCard.css";

function ResultCard({ result, originalImage }) {
  if (!result) return null;

  return (
    <div className="result-card-view animate-fade-in">
      
      {/* Dual Telemetry Visual Comparison Section */}
      <div className="segmentation-telemetry-panel">
        <div className="telemetry-box">
          <h3>Target Source Profile</h3>
          <div className="img-frame">
            <img src={originalImage} alt="Input source analysis target" />
          </div>
        </div>

        <div className="telemetry-box animate-pulse-border">
          <h3>UNet Isolated Mask</h3>
          {result.generated_mask ? (
            <div className="img-frame binary-mask-bg">
              <img src={result.generated_mask} alt="Segmented binary mask silhouette array" />
            </div>
          ) : (
            <div className="mask-fallback-error">
              <p>Mask rendering buffer parameter mismatch.</p>
            </div>
          )}
        </div>
      </div>

      <h2>Extracted Morphological Telemetry</h2>
      <div className="features-grid">
        <div className="feature-box">
          <h4>Max Length</h4>
          <p>{Number(result.length_cm).toFixed(2)} <span className="unit">cm</span></p>
        </div>

        <div className="feature-box">
          <h4>Max Thickness</h4>
          <p>{Number(result.thickness_cm).toFixed(2)} <span className="unit">cm</span></p>
        </div>

        <div className="feature-box">
          <h4>Contour Perimeter</h4>
          <p>{Number(result.perimeter_cm).toFixed(2)} <span className="unit">cm</span></p>
        </div>

        <div className="feature-box">
          <h4>Surface Area</h4>
          <p>{Number(result.area_cm2).toFixed(2)} <span className="unit">cm²</span></p>
        </div>

        <div className="feature-box">
          <h4>Volume Mass Proxy</h4>
          <p>{Number(result.volume_proxy_cm3).toFixed(2)} <span className="unit">cm³</span></p>
        </div>
      </div>

      {/* Main Prediction Presentation Box */}
      <div className="weight-box-hero">
        <h3>Calculated Mass Estimation</h3>
        <div className="weight-value-container">
          <span className="weight-number">{Number(result.predicted_weight_g).toFixed(2)}</span>
          <span className="weight-unit">grams</span>
        </div>
      </div>
    </div>
  );
}

export default ResultCard;