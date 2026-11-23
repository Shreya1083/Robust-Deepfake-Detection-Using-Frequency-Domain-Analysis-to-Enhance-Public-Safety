# 🔍 Robust Deepfake Detection Using Frequency Domain Analysis to Enhance Public Safety

A deep-learning application that detects deepfake images and videos by fusing **CNN spatial features** with **FFT** and **DWT frequency-domain features**, delivering accurate classification with visual evidence.

---

## 📋 Project Overview

Deepfakes pose a growing threat to public trust and safety. This project builds a robust detection system that goes beyond pixel-level analysis by examining **frequency-domain artifacts** introduced during the deepfake generation process.

### Key Capabilities

| #  | Feature                              | Description                                                  |
|----|--------------------------------------|--------------------------------------------------------------|
| 1  | Image-based detection                | Upload a single image for real / deepfake classification.    |
| 2  | Video-based detection                | Frame-by-frame analysis of uploaded video files.             |
| 3  | Face extraction                      | Automatic facial-region detection and cropping (OpenCV).     |
| 4  | CNN spatial features                 | Convolutional network captures spatial manipulation cues.    |
| 5  | FFT frequency features               | Fast Fourier Transform reveals frequency-domain anomalies.   |
| 6  | DWT wavelet features                 | Discrete Wavelet Transform provides multi-resolution detail. |
| 7  | Feature fusion (CNN + FFT + DWT)     | Multi-domain fusion for stronger classification.             |
| 8  | REAL / DEEPFAKE classification       | Binary classification with confidence score.                 |
| 9  | Frequency-based visual evidence      | Spectrum and wavelet visualizations for explainability.      |
| 10 | Robustness testing                   | Stress tests under compression, blur, noise & resizing.     |

---

## 🛠️ Technology Stack

| Category           | Technology       |
|--------------------|------------------|
| Deep Learning      | PyTorch          |
| Web UI             | Streamlit        |
| Computer Vision    | OpenCV           |
| Scientific Compute | NumPy, SciPy     |
| Wavelets           | PyWavelets       |
| Visualization      | Matplotlib, Seaborn |
| ML Utilities       | Scikit-learn     |

---

## 📁 Project Structure

```
Robust-Deepfake-Detection/
│
├── app.py                  # Streamlit application entry point
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation (this file)
├── .gitignore              # Git ignore rules
│
├── assets/
│   ├── images/             # UI images and icons
│   └── samples/            # Sample media for demo purposes
│
├── data/
│   ├── raw/                # Original unprocessed datasets
│   ├── processed/          # Preprocessed face-cropped images
│   ├── splits/             # Train / val / test split metadata
│   └── README.md           # Data directory documentation
│
├── src/
│   ├── __init__.py         # Package initializer
│   ├── config.py           # Centralized paths and settings
│   ├── preprocessing.py    # Image / video preprocessing
│   ├── face_extraction.py  # Facial region detection & cropping
│   ├── detector.py         # CNN spatial feature extractor
│   ├── frequency.py        # FFT frequency feature extraction
│   ├── wavelet.py          # DWT wavelet feature extraction
│   ├── feature_fusion.py   # CNN + FFT + DWT fusion classifier
│   ├── video_processor.py  # Video frame-level analysis
│   ├── robustness.py       # Robustness stress testing
│   └── visualization.py    # Frequency-domain visualizations
│
├── models/
│   └── checkpoints/        # Saved model weights (.pth)
│
├── pages/                  # Streamlit multi-page app pages
├── scripts/                # Utility and automation scripts
├── notebooks/              # Jupyter notebooks for exploration
└── tests/                  # Unit and integration tests
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Robust-Deepfake-Detection.git
cd Robust-Deepfake-Detection
```

### 2. Create a Virtual Environment

```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Application

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your default browser.

---

## 📌 Development Phases

| Phase | Milestone                                     | Status      |
|-------|-----------------------------------------------|-------------|
| 0     | Project scaffold and environment setup        | ✅ Complete |
| 1     | Face extraction and preprocessing pipeline    | ⬜ Planned  |
| 2     | FFT and DWT frequency feature extraction      | ⬜ Planned  |
| 3     | CNN spatial feature extractor                 | ⬜ Planned  |
| 4     | Feature fusion model and training             | ⬜ Planned  |
| 5     | Streamlit UI — image and video analysis       | ⬜ Planned  |
| 6     | Robustness testing suite                      | ⬜ Planned  |
| 7     | Final integration, evaluation and deployment  | ⬜ Planned  |

---

## 📄 License

This project is developed as a college mini project for academic purposes.

---

## 🙏 Acknowledgements

- **FaceForensics++** and related deepfake datasets for research benchmarks.
- Open-source libraries: PyTorch, Streamlit, OpenCV, PyWavelets.
