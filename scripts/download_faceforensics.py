"""
Official FaceForensics++ Dataset Downloader Reference & Interface
=================================================================
FaceForensics++ is hosted by the Visual Computing Lab at the Technical
University of Munich (TUM).

Access Requirements & Terms of Use:
-----------------------------------
1. In accordance with dataset terms, users must submit an official request
   form to TUM researchers via:
   https://github.com/ondyari/FaceForensics
2. Once granted access, the official download script (`download-faceforensics.py`)
   is provided along with download credentials.

Supported Subsets & Compressions:
---------------------------------
- Datasets:
    * original (Pristine YouTube sequences)
    * Deepfakes (DeepFake face swaps)
    * Face2Face (Facial expression reenactment)
    * FaceSwap (Graphics-based face swapping)
    * NeuralTextures (Rendering reenactment via neural textures)
- Compression Quality:
    * c23: Light compression (standard H.264 high-quality benchmark)
    * c40: Heavy compression (low quality)
    * raw: Uncompressed video sequences

Usage:
------
    python scripts/download_faceforensics.py <output_directory> -d all -c c23 -t videos
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FF_SUBSETS = ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
FF_COMPRESSIONS = ["c23", "c40", "raw"]


def print_official_instructions():
    print("""
===========================================================================
FaceForensics++ Official Access & Download Instructions
===========================================================================
To download the complete FaceForensics++ video dataset:
1. Navigate to: https://github.com/ondyari/FaceForensics
2. Complete the Google Form application for academic/research access.
3. Once approved, use the official downloader:
   python download-faceforensics.py <output_directory> -d all -c c23 -t videos

In VeriFrame:
A clean, verified FaceForensics++ C23 benchmark subset is prepared in:
  data/processed/real_faces
  data/processed/fake_faces
  data/splits/train.csv, validation.csv, test.csv
===========================================================================
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaceForensics++ downloader helper")
    parser.add_argument("--info", action="store_true", help="Display download terms & instructions")
    args = parser.parse_args()
    print_official_instructions()
