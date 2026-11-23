# 📊 Phase 2: Dataset Collection & Preprocessing Report

**Project:** Robust Deepfake Detection Using Frequency Domain Analysis to Enhance Public Safety

---

## 1. Raw Dataset Video Inventory

| Dataset Source | REAL Videos | FAKE Videos | Total Valid Videos | Corrupted / Invalid |
|---|---|---|---|---|
| **FaceForensics++** | 0 | 0 | 0 | 0 |
| **Celeb-DF (v2)** | 0 | 0 | 0 | 0 |
| **Overall Total** | **0** | **0** | **0** | **0** |

---

## 2. Frame Extraction & Face Detection

| Metric | Count |
|---|---|
| **Total Frames Extracted** | 0 (Real: 0, Fake: 0) |
| **Detected & Cropped Faces** | 0 (Real: 0, Fake: 0) |
| **Failed Extractions (No Face / Blurry)** | 0 |
| **Face Resolution** | 256 × 256 (RGB, normalized) |

---

## 3. Dataset Splits (Partitioned by Video ID)

*Data Leakage Prevention: Zero video ID overlap between splits.*

| Split | Partition Type | Unique Videos | REAL Faces | FAKE Faces | Total Images |
|---|---|---|---|---|---|
| **Train (70%)** | FaceForensics++ | 0 | 0 | 0 | **0** |
| **Validation (15%)** | FaceForensics++ | 0 | 0 | 0 | **0** |
| **Test (15%)** | FaceForensics++ | 0 | 0 | 0 | **0** |
| **Celeb-DF Eval (100%)** | Celeb-DF v2 (Cross-Dataset) | 0 | 0 | 0 | **0** |

---

## 4. Visual Evidence & Charts

Distribution plot generated: `data/dataset_distribution.png`
