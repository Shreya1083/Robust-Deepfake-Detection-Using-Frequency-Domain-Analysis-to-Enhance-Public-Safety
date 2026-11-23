import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Home from './pages/Home'
import About from './pages/About'
import ImageDetection from './pages/ImageDetection'
import VideoDetection from './pages/VideoDetection'
import FrequencyAnalysis from './pages/FrequencyAnalysis'
import Dashboard from './pages/Dashboard'

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-wrapper">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
            <Route path="/image-detection" element={<ImageDetection />} />
            <Route path="/video-detection" element={<VideoDetection />} />
            <Route path="/frequency-analysis" element={<FrequencyAnalysis />} />
            <Route path="/dashboard" element={<Dashboard />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  )
}
