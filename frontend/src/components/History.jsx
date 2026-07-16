import { useState, useEffect } from "react";
import "../styles/History.css";

function History() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  
  // Filtering state
  const [searchName, setSearchName] = useState("");
  const [sortWeight, setSortWeight] = useState(""); // "heavy", "light"
  const [sortDate, setSortDate] = useState("newest"); // "newest", "oldest"

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await fetch("http://127.0.0.1:8000/history");
        if (!response.ok) {
          throw new Error("Failed to fetch history");
        }
        const data = await response.json();
        setRecords(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  const getFilteredRecords = () => {
    let filtered = [...records];
    
    if (searchName) {
      filtered = filtered.filter(r => 
        r.original_filename.toLowerCase().includes(searchName.toLowerCase())
      );
    }
    
    if (sortWeight === "heavy") {
      filtered.sort((a, b) => b.predicted_weight_g - a.predicted_weight_g);
    } else if (sortWeight === "light") {
      filtered.sort((a, b) => a.predicted_weight_g - b.predicted_weight_g);
    } else {
      // sort by date
      if (sortDate === "newest") {
        filtered.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      } else {
        filtered.sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
      }
    }
    
    return filtered;
  };

  const displayRecords = getFilteredRecords();

  return (
    <div className="history-container">
      <div className="history-header">
        <h2>Your Prediction History</h2>
        
        <div className="filters-container">
          <input 
            type="text" 
            placeholder="Search by filename..." 
            value={searchName}
            onChange={(e) => setSearchName(e.target.value)}
            className="search-input"
          />
          
          <select 
            value={sortWeight} 
            onChange={(e) => setSortWeight(e.target.value)}
            className="sort-select"
          >
            <option value="">Sort by Weight (Default)</option>
            <option value="heavy">Heaviest to Lightest</option>
            <option value="light">Lightest to Heaviest</option>
          </select>
          
          <select 
            value={sortDate} 
            onChange={(e) => setSortDate(e.target.value)}
            className="sort-select"
          >
            <option value="newest">Newest First</option>
            <option value="oldest">Oldest First</option>
          </select>
        </div>
      </div>
      
      {loading && <p className="history-msg loading">Loading history...</p>}
      {error && <p className="history-msg error">{error}</p>}
      
      {!loading && !error && displayRecords.length === 0 && (
        <div className="history-msg empty">
          <p>No records found matching your filters.</p>
        </div>
      )}
      
      <div className="history-grid">
        {displayRecords.map((record) => (
          <div key={record.id} className="history-card">
            <div className="history-card-images">
              <div className="history-img-wrapper">
                <span className="badge">Original</span>
                <img src={`http://127.0.0.1:8000${record.image_path}`} alt="Original uploaded fish" />
              </div>
              <div className="history-img-wrapper">
                <span className="badge">Mask</span>
                <img src={`http://127.0.0.1:8000${record.mask_path}`} alt="Generated segmentation mask" />
              </div>
            </div>
            
            <div className="history-card-content">
              <h3 className="file-name">{record.original_filename}</h3>
              <p className="timestamp">{new Date(record.created_at).toLocaleString()}</p>
              
              <div className="measurement-stats">
                <div className="stat-highlight">
                  <span className="stat-label">Predicted Weight</span>
                  <span className="stat-value highlight">{record.predicted_weight_g.toFixed(2)} g</span>
                </div>
                
                <div className="metrics-grid">
                  <div className="metric">
                    <span>Length:</span> <strong>{record.length_cm.toFixed(2)} cm</strong>
                  </div>
                  <div className="metric">
                    <span>Thickness:</span> <strong>{record.thickness_cm.toFixed(2)} cm</strong>
                  </div>
                  <div className="metric">
                    <span>Area:</span> <strong>{record.area_cm2.toFixed(2)} cm²</strong>
                  </div>
                  <div className="metric">
                    <span>Volume Proxy:</span> <strong>{record.volume_proxy_cm3.toFixed(2)} cm³</strong>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default History;
