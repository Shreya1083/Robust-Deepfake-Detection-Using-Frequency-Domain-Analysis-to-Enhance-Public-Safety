"""
Centralized Configuration
=========================
All project-wide paths, constants and default settings live here so
that every other module imports them from a single source of truth.
"""

from pathlib import Path

# ============================================================
# Project Metadata
# ============================================================

PROJECT_TITLE = "Robust Deepfake Detection Using Frequency Domain Analysis to Enhance Public Safety"
PROJECT_SHORT_TITLE = "Robust Deepfake Detection"
PROJECT_SUBTITLE = "Using Frequency Domain Analysis to Enhance Public Safety"

# ============================================================
# Directory Paths
# ============================================================

# Project root (two levels up from this file: src/config.py → root)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
INTERIM_DATA_DIR = DATA_DIR / "interim"

# Raw dataset subdirectories
FACEFORENSICS_DIR = RAW_DATA_DIR / "faceforensics"
FF_REAL_DIR = FACEFORENSICS_DIR / "real"
FF_FAKE_DIR = FACEFORENSICS_DIR / "fake"

CELEB_DF_DIR = RAW_DATA_DIR / "celeb_df"
CDF_REAL_DIR = CELEB_DF_DIR / "real"
CDF_FAKE_DIR = CELEB_DF_DIR / "fake"

# Processed face crop subdirectories
PROCESSED_REAL_FACES_DIR = PROCESSED_DATA_DIR / "real_faces"
PROCESSED_FAKE_FACES_DIR = PROCESSED_DATA_DIR / "fake_faces"

# Interim frames directories
INTERIM_FRAMES_DIR = INTERIM_DATA_DIR / "frames"
INTERIM_REAL_FRAMES_DIR = INTERIM_FRAMES_DIR / "real"
INTERIM_FAKE_FRAMES_DIR = INTERIM_FRAMES_DIR / "fake"

# Dataset split CSV files
TRAIN_SPLIT_FILE = SPLITS_DIR / "train.csv"
VAL_SPLIT_FILE = SPLITS_DIR / "validation.csv"
TEST_SPLIT_FILE = SPLITS_DIR / "test.csv"
CELEB_DF_EVAL_FILE = SPLITS_DIR / "celeb_df_eval.csv"

# Model artefacts
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Static assets
ASSETS_DIR = ROOT_DIR / "assets"
IMAGES_DIR = ASSETS_DIR / "images"
SAMPLES_DIR = ASSETS_DIR / "samples"

# ============================================================
# Video & Image Processing Settings
# ============================================================

SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".m4v", ".mkv")
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Frame sampling defaults
FRAME_SAMPLE_RATE = 10         # Extract every 10th frame
MAX_FRAMES_PER_VIDEO = 20      # Limit frames per video to prevent redundancy

# Face extraction settings
FACE_PADDING_RATIO = 0.2       # 20% margin around bounding box to capture hairline/edges
FACE_MIN_SIZE = (64, 64)       # Discard detections smaller than 64x64
FACE_DETECTION_CONFIDENCE = 0.5  # Minimum confidence threshold

IMAGE_SIZE = (256, 256)        # (height, width) standard input for deep learning models

# ============================================================
# Model Settings (placeholders — populated in later phases)
# ============================================================

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
NUM_CLASSES = 2                  # REAL, DEEPFAKE

# ============================================================
# Class Labels
# ============================================================

CLASS_LABELS = {0: "REAL", 1: "DEEPFAKE"}

# ============================================================
# Robustness Test Defaults
# ============================================================

COMPRESSION_QUALITIES = [95, 75, 50, 30, 10]
BLUR_KERNELS = [3, 5, 7, 9]
NOISE_LEVELS = [5, 10, 20, 40]
RESIZE_FACTORS = [0.25, 0.5, 0.75, 1.5, 2.0]
