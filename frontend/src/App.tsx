import { BrowserRouter, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import ProcessPage from './pages/ProcessPage';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/process" element={<ProcessPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
