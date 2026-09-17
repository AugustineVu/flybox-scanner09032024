import unittest

import cv2
import numpy

from detection.motion import MotionDetector
from utils.app_settings import AppSettings


class TestMotionDetector(unittest.TestCase):
    def setUp(self):
        self.settings = AppSettings(keep_defaults=True)
        self.detector = MotionDetector(self.settings)

    def test_detect(self):
        # feed 20 real frames to the subtractor to build a model
        for i in range(20):
            frame = cv2.imread(f"tests/fixtures/frames/{i + 1}.jpg")
            self.detector.detect_with_bg_subtractor(frame)

        contours = self.detector.detect_with_bg_subtractor(frame)
        self.assertEqual(len(contours), 39)


class TestContourFiltering(unittest.TestCase):
    # the subtractor emits single pixel specks, and a mask covering nearly the whole
    # frame each time it reinitialises on a lighting change. neither is a fly
    def setUp(self):
        self.detector = MotionDetector(AppSettings(keep_defaults=True))

    def make_mask(self, *, speck=False, fly=False, flash=False):
        mask = numpy.zeros((100, 100), dtype=numpy.uint8)
        if speck:
            mask[20, 20] = 255
        if fly:
            cv2.rectangle(mask, (50, 10), (56, 16), 255, -1)
        if flash:
            cv2.rectangle(mask, (0, 60), (99, 99), 255, -1)
        return mask

    def test_keeps_a_fly_sized_contour(self):
        contours = self.detector.find_contours(self.make_mask(fly=True))

        self.assertEqual(len(contours), 1)
        self.assertGreater(cv2.contourArea(contours[0]), 0)

    def test_drops_a_single_pixel_speck(self):
        # contourArea is 0 here, which means it has no usable centre at all
        self.assertEqual(cv2.contourArea(
            cv2.findContours(self.make_mask(speck=True), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)[0][0]), 0)

        self.assertEqual(self.detector.find_contours(self.make_mask(speck=True)), [])

    def test_drops_a_full_frame_reinit_flash(self):
        self.assertEqual(self.detector.find_contours(self.make_mask(flash=True)), [])

    def test_keeps_the_fly_and_drops_the_rest(self):
        mask = self.make_mask(speck=True, fly=True, flash=True)

        contours = self.detector.find_contours(mask)

        self.assertEqual(len(contours), 1)
        self.assertEqual(cv2.contourArea(contours[0]), 36)
