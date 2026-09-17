import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np

from components.frame_canvas import FrameCanvas
from components.scan_canvas import ScanCanvas
from detection.grids import GridDetector

# ScanCanvas needs a Tk window and a capture device to instantiate, so these tests
# call its methods directly with a stand-in self. that's enough to pin the thing that
# actually went wrong: which frame detection runs on

GRID_FIXTURE = "tests/fixtures/grid.jpg"


def canvas_with(grid):
    canvas = MagicMock()
    canvas.grid = grid
    return canvas


class TestScanCanvasOverlay(unittest.TestCase):
    def setUp(self):
        self.image = cv2.imread(GRID_FIXTURE)
        self.grid = GridDetector(self.image).detect()

    def test_frame_acquisition_is_not_overridden(self):
        # the overlay used to be drawn inside get_frame, which meant detect_grid ran
        # detection over the rectangles the previous scan had drawn. acquisition has
        # to stay untouched so that detection and display can share it safely
        self.assertIs(ScanCanvas.get_frame, FrameCanvas.get_frame)

    def test_draw_grid_does_nothing_before_a_scan(self):
        frame = self.image.copy()

        ScanCanvas.draw_grid(canvas_with(None), frame)

        self.assertTrue(np.array_equal(frame, self.image))

    def test_draw_grid_marks_the_frame_once_there_is_a_grid(self):
        frame = self.image.copy()

        ScanCanvas.draw_grid(canvas_with(self.grid), frame)

        self.assertFalse(np.array_equal(frame, self.image))

    def test_the_overlay_would_corrupt_a_rescan(self):
        # this is why the overlay has to stay out of the detection path: feeding an
        # annotated frame back into detection moves the wells and inflates the radius
        annotated = self.image.copy()
        ScanCanvas.draw_grid(canvas_with(self.grid), annotated)

        clean_detector = GridDetector(self.image)
        clean_detector.detect()
        annotated_detector = GridDetector(annotated)
        annotated_detector.detect()

        self.assertNotAlmostEqual(
            clean_detector.average_radius,
            annotated_detector.average_radius,
            places=1,
        )
