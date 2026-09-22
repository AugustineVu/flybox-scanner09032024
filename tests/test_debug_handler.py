import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from detection.grids import GridDetector
from handlers.debug import DebugHandler, DebugOptions
from handlers.frame import FrameHandler
from utils.app_settings import AppSettings

GRID_FIXTURE = "tests/fixtures/grid.jpg"


class IndexOverlayOff(DebugOptions):
    # the state the handler could not previously be constructed in
    def __init__(self):
        super().__init__()
        self.draw_index = False


class TestDebugHandler(unittest.TestCase):
    def setUp(self):
        self.grid = GridDetector(cv2.imread(GRID_FIXTURE)).detect()
        self.inner = MagicMock()
        self.handler = DebugHandler(self.grid, self.inner)
        self.frame = np.zeros((411, 640, 3), dtype=np.uint8)

    def test_satisfies_the_motion_event_handler_contract(self):
        # FrameHandler reads on_frame off its handler on every single frame
        self.assertTrue(callable(self.handler.on_frame))

    def test_on_frame_exists_even_when_the_index_overlay_starts_off(self):
        with patch("handlers.debug.DebugOptions", IndexOverlayOff):
            handler = DebugHandler(self.grid, self.inner)

        self.assertTrue(callable(handler.on_frame))

    def test_the_overlay_can_be_turned_on_after_starting_off(self):
        # nothing was listening for the option before, so toggling it did nothing
        with patch("handlers.debug.DebugOptions", IndexOverlayOff):
            handler = DebugHandler(self.grid, self.inner)
        handler.on_frame(self.frame)
        self.assertTrue(np.array_equal(self.frame, np.zeros_like(self.frame)))

        handler.options.toggle("draw_index")
        handler.on_frame(self.frame)

        self.assertFalse(np.array_equal(self.frame, np.zeros_like(self.frame)))

    def test_draws_well_indices_by_default(self):
        self.handler.on_frame(self.frame)

        self.assertFalse(np.array_equal(self.frame, np.zeros_like(self.frame)))

    def test_draws_nothing_once_the_index_option_is_off(self):
        self.handler.options.toggle("draw_index")

        self.handler.on_frame(self.frame)

        self.assertTrue(np.array_equal(self.frame, np.zeros_like(self.frame)))

    def make_event(self):
        # the overlays draw straight onto the event, so it needs real pixels
        event = MagicMock()
        event.frame = np.zeros((80, 80, 3), dtype=np.uint8)
        contour = np.array(
            [[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32
        )
        event.point.contour = contour
        event.last_point.contour = contour
        event.point.center = (15.0, 15.0)
        event.last_point.center = (25.0, 25.0)
        event.point.item.bounds = ((5.0, 5.0), (45.0, 45.0))
        return event

    def test_passes_events_through_to_the_wrapped_handler(self):
        event = self.make_event()

        self.handler.handle(event)

        self.inner.handle.assert_called_once_with(event)

    def test_draws_on_the_frame_the_event_carries(self):
        event = self.make_event()

        self.handler.handle(event)

        self.assertFalse(np.array_equal(event.frame, np.zeros_like(event.frame)))

    def test_draws_nothing_while_hidden(self):
        event = self.make_event()
        self.handler.options.toggle("hidden")

        self.handler.handle(event)

        self.inner.handle.assert_called_once_with(event)
        self.assertTrue(np.array_equal(event.frame, np.zeros_like(event.frame)))


class TestDebugHandlerDrivenByFrameHandler(unittest.TestCase):
    # the combination the app actually builds, and the one that used to break
    def test_a_frame_handler_can_drive_a_debug_handler(self):
        grid = GridDetector(cv2.imread(GRID_FIXTURE)).detect()
        with patch("handlers.debug.DebugOptions", IndexOverlayOff):
            debug_handler = DebugHandler(grid, MagicMock())
        frame_handler = FrameHandler(grid, AppSettings(keep_defaults=True), debug_handler)

        for i in range(3):
            frame = cv2.imread(f"tests/fixtures/frames/{i + 1}.jpg")
            frame_handler.handle(frame, i + 1)
