import { Link } from "react-router-dom";
import "../styles/Navbar.css";

function Navbar() {
  return (
    <nav className="navbar">
      <h1><Link to="/" className="logo-link">Fish Weight Estimator</Link></h1>
      <div className="nav-links">
        <Link to="/" className="nav-item">Upload</Link>
        <Link to="/history" className="nav-item">History</Link>
      </div>
    </nav>
  );
}

export default Navbar;