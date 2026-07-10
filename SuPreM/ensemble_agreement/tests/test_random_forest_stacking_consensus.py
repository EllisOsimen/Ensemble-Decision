from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import random_forest_stacking_consensus as stacking  # noqa: E402


class RandomForestStackingLabelTests(unittest.TestCase):
    def test_load_label_image_rounds_scaled_nifti_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "scaled_labels.nii.gz"
            expected = np.array([0, 1, 2, 3], dtype=np.float64).reshape(1, 1, 4)
            image = nib.Nifti1Image(expected, np.eye(4))
            image.set_data_dtype(np.int16)
            nib.save(image, str(path))

            decoded = np.asanyarray(nib.load(str(path)).dataobj)
            self.assertEqual(decoded.astype(np.int16).ravel().tolist(), [0, 0, 1, 2])

            labels, _ = stacking.load_label_image(path)

            np.testing.assert_array_equal(labels, expected.astype(np.int16))

    def test_load_label_image_rejects_non_integer_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "invalid_labels.nii.gz"
            image = nib.Nifti1Image(np.array([0.0, 1.25]).reshape(1, 1, 2), np.eye(4))
            nib.save(image, str(path))

            with self.assertRaisesRegex(ValueError, "non-integer label values"):
                stacking.load_label_image(path)

    def test_training_requires_every_curvas_class(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"missing expected classes \[3\]"):
            stacking.validate_training_class_coverage(np.array([0, 1, 2], dtype=np.uint8))

    def test_training_accepts_all_curvas_classes(self) -> None:
        stacking.validate_training_class_coverage(np.array([0, 1, 2, 3], dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
