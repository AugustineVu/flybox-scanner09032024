import unittest
from unittest.mock import MagicMock, patch

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


def scan_canvas_with(grid, expected=(8, 12)):
    canvas = MagicMock()
    canvas.grid = grid
    canvas.window.settings.get.side_effect = lambda key: {
        "grid.rows": expected[0],
        "grid.columns": expected[1],
    }[key]
    return canvas


class TestScanCanvasCanRecord(unittest.TestCase):
    def setUp(self):
        self.grid = GridDetector(cv2.imread(GRID_FIXTURE)).detect()

    def test_a_good_scan_can_record(self):
        self.assertTrue(ScanCanvas.can_record(scan_canvas_with(self.grid)))

    def test_no_grid_cannot_record(self):
        # detection failing used to leave the previous grid in place
        self.assertFalse(ScanCanvas.can_record(scan_canvas_with(None)))

    def test_a_grid_of_the_wrong_shape_cannot_record(self):
        self.assertFalse(
            ScanCanvas.can_record(scan_canvas_with(self.grid, expected=(6, 16)))
        )


class TestScanCanvasStartRecording(unittest.TestCase):
    # --tuning motion calls start_recording directly, skipping the Record button,
    # so the button being disabled is not enough to stop a bad grid getting through
    def test_refuses_to_start_without_a_usable_grid(self):
        canvas = MagicMock()
        canvas.can_record.return_value = False

        with patch("components.scan_canvas.messagebox") as messagebox:
            ScanCanvas.start_recording(canvas)

        canvas.window.state_manager.record.assert_not_called()
        self.assertNotIn("grid", canvas.window.app_state)
        messagebox.showwarning.assert_called_once()

    def test_starts_recording_when_the_scan_is_good(self):
        canvas = MagicMock()
        canvas.can_record.return_value = True
        canvas.window.app_state = {}

        ScanCanvas.start_recording(canvas)

        canvas.window.state_manager.record.assert_called_once()
        self.assertIs(canvas.window.app_state["grid"], canvas.grid)
