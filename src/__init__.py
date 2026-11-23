"""
src — Core package for Robust Deepfake Detection.

Modules:
    config          Centralized paths and project-wide settings.
    preprocessing   Image and video preprocessing pipelines.
    face_extraction Facial region detection and cropping.
    detector        CNN-based spatial feature extractor.
    frequency       FFT-based frequency feature extraction.
    wavelet         DWT-based wavelet feature extraction.
    feature_fusion  Multi-domain feature fusion and classifier.
    video_processor Frame-level video analysis pipeline.
    robustness      Compression, blur, noise and resize testing.
    visualization   Frequency-domain visual evidence generation.
"""
