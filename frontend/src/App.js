import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import UploadForm from "./components/UploadForm";
import History from "./components/History";
import "./styles/App.css";

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<UploadForm />} />
        <Route path="/history" element={<History />} />
      </Routes>
    </Router>
  );
}

export default App;
