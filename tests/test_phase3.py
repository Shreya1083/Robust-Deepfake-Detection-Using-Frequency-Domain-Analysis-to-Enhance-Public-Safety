"""
Phase 3 Unit and Integration Tests
==================================
Tests ResNet18 spatial model architecture, feature extraction, dataset loading,
training loop logic, checkpointing, and detector inference.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.spatial_model import SpatialResNet18, build_spatial_model, get_default_device
from scripts.train_model import train_one_epoch, validate
from src.dataset_loader import DeepfakeFaceDataset, get_spatial_transforms
from src.detector import SpatialDetector


class TestSpatialModel(unittest.TestCase):
    """Test ResNet18 spatial model architecture and feature extraction."""

    def setUp(self):
        self.device = get_default_device()
        self.model, _ = build_spatial_model(pretrained=True, device=self.device)

    def test_forward_output_shape(self):
        # Batch of 2 RGB images (224x224)
        dummy_input = torch.randn(2, 3, 224, 224).to(self.device)
        logits = self.model(dummy_input)

        self.assertEqual(logits.shape, (2, 2))

    def test_extract_features_shape(self):
        # Must return 512-dim embedding for Phase 6 feature fusion
        dummy_input = torch.randn(3, 3, 224, 224).to(self.device)
        features = self.model.extract_features(dummy_input)

        self.assertEqual(features.shape, (3, 512))

    def test_freeze_unfreeze_backbone(self):
        # Test freeze
        self.model.freeze_backbone_layers(freeze=True)
        backbone_grads = [p.requires_grad for p in self.model.backbone.parameters()]
        classifier_grads = [p.requires_grad for p in self.model.classifier.parameters()]

        self.assertTrue(all(not g for g in backbone_grads))
        self.assertTrue(all(g for g in classifier_grads))

        # Test unfreeze
        self.model.freeze_backbone_layers(freeze=False)
        backbone_grads_unfrozen = [p.requires_grad for p in self.model.backbone.parameters()]
        self.assertTrue(all(g for g in backbone_grads_unfrozen))

    def test_predict_proba_sums_to_one(self):
        dummy_input = torch.randn(4, 3, 224, 224).to(self.device)
        probs = self.model.predict_proba(dummy_input).cpu()

        self.assertEqual(probs.shape, (4, 2))
        sums = torch.sum(probs, dim=1).detach().numpy()
        np.testing.assert_allclose(sums, np.ones(4), atol=1e-5)


class TestTrainingAndCheckpointing(unittest.TestCase):
    """Test training loop step, validation, and checkpoint persistence."""

    def test_training_step_and_validation(self):
        device = torch.device("cpu")
        model = SpatialResNet18(pretrained=False, num_classes=2)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Synthetic mock dataset of 4 samples
        inputs = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        # 1. Run train epoch
        loss, acc = train_one_epoch(model, loader, criterion, optimizer, device)
        self.assertGreater(loss, 0.0)
        self.assertTrue(0.0 <= acc <= 1.0)

        # 2. Run validation
        val_loss, val_acc = validate(model, loader, criterion, device)
        self.assertGreater(val_loss, 0.0)
        self.assertTrue(0.0 <= val_acc <= 1.0)

    def test_checkpoint_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_checkpoint.pth"

            model = SpatialResNet18(pretrained=False, num_classes=2)
            checkpoint_data = {
                "epoch": 5,
                "model_state_dict": model.state_dict(),
                "val_loss": 0.25,
                "val_acc": 0.92,
                "architecture": "resnet18",
                "num_classes": 2,
                "class_mapping": {0: "REAL", 1: "DEEPFAKE"},
                "input_size": (224, 224),
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            }
            torch.save(checkpoint_data, ckpt_path)
            self.assertTrue(ckpt_path.exists())

            # Reload with SpatialDetector
            detector = SpatialDetector(checkpoint_path=ckpt_path)
            self.assertTrue(detector.is_ready)
            self.assertEqual(detector.checkpoint_meta["architecture"], "resnet18")
            self.assertEqual(detector.checkpoint_meta["class_mapping"][0], "REAL")


class TestDetectorInference(unittest.TestCase):
    """Test SpatialDetector inference and probability bounds."""

    def test_prediction_output_schema_and_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "detector_test.pth"

            # Create and save a valid initialized model
            model = SpatialResNet18(pretrained=False, num_classes=2)
            torch.save({"model_state_dict": model.state_dict(), "num_classes": 2}, ckpt_path)

            detector = SpatialDetector(checkpoint_path=ckpt_path)
            self.assertTrue(detector.is_ready)

            # Synthetic RGB test image
            test_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            result = detector.predict_image(test_img)

            # Check output fields
            self.assertTrue(result["is_ready"])
            self.assertIn(result["prediction"], ["REAL", "DEEPFAKE"])
            self.assertIn(result["label"], [0, 1])
            self.assertIsNone(result["error"])

            # Probability bounds
            self.assertTrue(0.0 <= result["real_prob"] <= 1.0)
            self.assertTrue(0.0 <= result["deepfake_prob"] <= 1.0)
            self.assertAlmostEqual(result["real_prob"] + result["deepfake_prob"], 1.0, places=3)
            self.assertEqual(result["confidence"], max(result["real_prob"], result["deepfake_prob"]))


if __name__ == "__main__":
    unittest.main()
