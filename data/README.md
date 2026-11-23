# Dataset Management Guide

This directory holds the raw and processed datasets for **Robust Deepfake Detection Using Frequency Domain Analysis to Enhance Public Safety**.

---

## 📁 Directory Structure

```
data/
├── raw/
│   ├── faceforensics/          # Primary training dataset (FaceForensics++)
│   │   ├── real/               # Pristine / authentic videos (e.g., original YouTube)
│   │   └── fake/               # Manipulated videos (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
│   └── celeb_df/               # Independent evaluation dataset (Celeb-DF v2)
│       ├── real/               # Pristine celebrity & YouTube videos (Celeb-real, YouTube-real)
│       └── fake/               # Synthesized deepfake videos (Celeb-synthesis)
│
├── interim/
│   └── frames/                 # Extracted video frames before face cropping
│       ├── real/
│       └── fake/
│
├── processed/
│   ├── real_faces/             # Cropped and normalized authentic face images
│   └── fake_faces/             # Cropped and normalized deepfake face images
│
└── splits/
    ├── train.csv               # Training set (video-level split)
    ├── validation.csv          # Validation set (video-level split)
    ├── test.csv                # Test set (video-level split)
    └── celeb_df_eval.csv       # Dedicated independent evaluation split
```

---

## 📥 How to Obtain and Place Datasets

Due to licensing terms and ethics guidelines, datasets cannot be redistributed directly inside this repository. Follow the official channels below to download them:

### 1. FaceForensics++ (Primary Training Dataset)

- **Official Repository**: [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)
- **Request Access**: Fill out the official [FaceForensics Terms of Use Google Form](https://forms.gle/4B9uWcL7gU1rS3P5A) to receive the official download script (`download-FaceForensics.py`).
- **Download Command** (example for c23 compressed videos):
  ```bash
  python download-FaceForensics.py <output_directory> -d Deepfakes -c c23 -t videos
  python download-FaceForensics.py <output_directory> -d original -c c23 -t videos
  ```
- **Placement**:
  - Place pristine/original videos into: `data/raw/faceforensics/real/`
  - Place manipulated videos into: `data/raw/faceforensics/fake/`
  *(The `dataset_loader.py` also automatically recognizes official subfolders if extracted directly into `data/raw/faceforensics/`)*

---

### 2. Celeb-DF (v2) (Independent Evaluation Dataset)

- **Official Website**: [https://github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics)
- **Request Access**: Submit the official dataset request form to the dataset maintainers (University at Albany, SUNY).
- **Placement**:
  - Place authentic videos (`Celeb-real`, `YouTube-real`) into: `data/raw/celeb_df/real/`
  - Place synthesized videos (`Celeb-synthesis`) into: `data/raw/celeb_df/fake/`

---

## ⚙️ Running the Data Preparation Pipeline

Once you have placed your downloaded video files in the raw directories:

1. **Extract Frames**:
   ```powershell
   .\.venv\Scripts\python.exe scripts/extract_frames.py --sample-rate 10 --max-frames 20
   ```

2. **Detect Faces, Preprocess & Split Dataset**:
   ```powershell
   .\.venv\Scripts\python.exe scripts/prepare_dataset.py
   ```

3. **Generate Dataset Analysis & Distribution Plots**:
   ```powershell
   .\.venv\Scripts\python.exe scripts/analyze_dataset.py
   ```

---

## 🛡️ Data Leakage Prevention

All splits (`train.csv`, `validation.csv`, `test.csv`) are strictly split **by source video ID** rather than by individual frames. Multiple frames from the same video are guaranteed to reside in the **same split partition**, preventing model over-fitting and false performance inflation.
Celeb-DF v2 is kept completely separate as an unseen cross-dataset evaluation benchmark.
