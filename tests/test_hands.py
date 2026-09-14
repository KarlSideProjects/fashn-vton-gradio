import unittest
import numpy as np
from PIL import Image
from tryon.hands import restore_hands, HandRepairError

class HandTests(unittest.TestCase):
    def setUp(self):
        self.source = Image.new('RGB', (100, 100), 'red')
        self.generated = Image.new('RGB', (100, 100), 'blue')
        self.labels = np.zeros((100, 100), dtype=np.uint8)
        self.labels[30:60, 40:60] = 13

    def test_restore_core_without_changing_background(self):
        image, report = restore_hands(self.source, self.generated, self.labels, self.labels)
        self.assertEqual(image.getpixel((50, 45)), (255, 0, 0))
        self.assertEqual(image.getpixel((10, 10)), (0, 0, 255))
        self.assertEqual(report['status'], 'restored')

    def test_new_or_missing_hands_are_rejected(self):
        for a,b in [(self.labels, np.zeros_like(self.labels)), (np.zeros_like(self.labels), self.labels)]:
            with self.subTest(), self.assertRaises(HandRepairError):
                restore_hands(self.source, self.generated, a, b)

    def test_shifted_hand_is_not_pasted(self):
        shifted = np.roll(self.labels, 30, axis=1)
        with self.assertRaises(HandRepairError):
            restore_hands(self.source, self.generated, self.labels, shifted)

    def test_sleeve_occlusion_is_rejected(self):
        covered = self.labels.copy(); covered[30:40,40:60] = 3
        with self.assertRaises(HandRepairError):
            restore_hands(self.source, self.generated, self.labels, covered)

    def test_extra_finger_outside_source_is_rejected(self):
        extra = self.labels.copy(); extra[40:50,60:85] = 13
        with self.assertRaises(HandRepairError):
            restore_hands(self.source, self.generated, self.labels, extra)

    def test_invalid_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            restore_hands(self.source, self.generated, self.labels[:50], self.labels)

    def test_no_detected_hand_does_not_claim_success(self):
        with self.assertRaises(HandRepairError):
            restore_hands(self.source, self.generated, self.labels*0, self.labels*0)

    def test_thin_fingers_are_copied_without_generated_pixels(self):
        labels = self.labels.copy()
        labels[20:30, 45:47] = 13
        image, _ = restore_hands(self.source, self.generated, labels, labels)
        np.testing.assert_array_equal(np.asarray(image)[labels == 13],
                                      np.asarray(self.source)[labels == 13])
        np.testing.assert_array_equal(np.asarray(image)[labels != 13],
                                      np.asarray(self.generated)[labels != 13])

    def test_missing_smaller_hand_is_not_hidden_by_larger_hand(self):
        labels = np.zeros_like(self.labels)
        labels[20:70, 20:50] = 13
        labels[40:46, 70:76] = 13
        generated_labels = labels.copy()
        generated_labels[40:46, 70:76] = 0
        with self.assertRaises(HandRepairError):
            restore_hands(self.source, self.generated, labels, generated_labels)

    def test_extra_small_hand_is_not_hidden_by_larger_hand(self):
        labels = np.zeros_like(self.labels)
        labels[20:70, 20:50] = 13
        generated_labels = labels.copy()
        generated_labels[40:46, 70:76] = 13
        with self.assertRaises(HandRepairError):
            restore_hands(self.source, self.generated, labels, generated_labels)

    def test_only_wrist_seam_is_feathered(self):
        labels = self.labels.copy()
        labels[60:70, 40:60] = 12  # Arm meeting the wrist.
        image, _ = restore_hands(self.source, self.generated, labels, labels)
        self.assertEqual(image.getpixel((40, 30)), (255, 0, 0))
        self.assertNotEqual(image.getpixel((50, 59)), (255, 0, 0))
        self.assertEqual(image.getpixel((50, 60)), (0, 0, 255))

    def test_two_hands_are_preserved_and_reported_individually(self):
        labels = self.labels.copy()
        labels[40:48, 75:83] = 13
        image, report = restore_hands(self.source, self.generated, labels, labels)
        self.assertEqual(len(report['regions']), 2)
        np.testing.assert_array_equal(np.asarray(image)[labels == 13],
                                      np.asarray(self.source)[labels == 13])

    def test_small_hand_sleeve_occlusion_is_checked_individually(self):
        labels = np.zeros_like(self.labels)
        labels[20:70, 20:50] = 13
        labels[40:50, 70:80] = 13
        covered = labels.copy()
        covered[40:41, 70:80] = 3
        with self.assertRaises(HandRepairError):
            restore_hands(self.source, self.generated, labels, covered)
