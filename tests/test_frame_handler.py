import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np

from detection.grids import GridDetector
from handlers.frame import FrameHandler
from utils.app_settings import AppSettings

# these tests run against a grid detected from the real fixture rather than a mock.
# a MagicMock grid hands back the same item object for every lookup, which collapses
# all 96 wells into one and hides anything to do with assigning flies to wells

GRID_FIXTURE = "tests/fixtures/grid.jpg"
FRAME_COUNT = 20


def make_grid():
    grid = GridDetector(cv2.imread(GRID_FIXTURE)).detect()
    return grid


def square_contour(center_x, center_y, size):
    half = size / 2
    return np.array(
        [
            [[center_x - half, center_y - half]],
            [[center_x + half, center_y - half]],
            [[center_x + half, center_y + half]],
            [[center_x - half, center_y + half]],
        ],
        dtype=np.int32,
    )


class TestFrameHandler(unittest.TestCase):
    def setUp(self):
        self.grid = make_grid()
        self.settings = AppSettings(keep_defaults=True)
        self.handler = MagicMock()
        self.frame_handler = FrameHandler(self.grid, self.settings, self.handler)

    def run_frames(self):
        for i in range(FRAME_COUNT):
            frame = cv2.imread(f"tests/fixtures/frames/{i + 1}.jpg")
            self.assertIsNotNone(frame)
            self.frame_handler.handle(frame, i + 1)
        return [call[0][0] for call in self.handler.handle.call_args_list]

    def test_fixture_grid_is_the_expected_shape(self):
        # if this fails the other tests are measuring the wrong thing
        self.assertEqual(self.grid.dimensions, (8, 12))
        self.assertTrue(self.grid.is_rectangular)

    def test_calls_on_frame_once_per_frame(self):
        self.run_frames()

        self.assertEqual(self.handler.on_frame.call_count, FRAME_COUNT)

    def test_emits_events(self):
        events = self.run_frames()

        self.assertGreater(len(events), 0)
        # the first frame has nothing to compare against, so it can't produce an event
        self.assertNotIn(1, [event.point.frame_count for event in events])

    def test_every_event_compares_against_an_earlier_frame(self):
        # a well that picks up two contours in one frame used to emit the gap between
        # them as movement, which made a still fly look like a moving one
        events = self.run_frames()

        for event in events:
            self.assertLess(
                event.last_point.frame_count,
                event.point.frame_count,
                "event measured movement within a single frame",
            )

    def test_at_most_one_event_per_well_per_frame(self):
        events = self.run_frames()

        seen = [(event.item.coords, event.point.frame_count) for event in events]
        self.assertCountEqual(seen, set(seen))

    def test_events_are_assigned_to_the_well_the_fly_is_in(self):
        events = self.run_frames()

        rows, columns = self.grid.dimensions
        for event in events:
            row_index, column_index = event.item.coords
            self.assertIn(row_index, range(rows))
            self.assertIn(column_index, range(columns))
            self.assertTrue(
                event.item.contains(event.point.center),
                f"well {event.item.coords} does not contain {event.point.center}",
            )

    def test_flies_do_not_travel_further_than_their_well(self):
        # a single fly cannot cross more than its own well between two frames,
        # so anything larger means we paired up points that aren't the same fly
        events = self.run_frames()

        (start_point, end_point) = self.grid.rows[0].items[0].bounds
        well_width = end_point[0] - start_point[0]
        for event in events:
            self.assertLess(event.distance, well_width)


class TestGridItemLookup(unittest.TestCase):
    # on a grid that isn't level, row bounding boxes overlap heavily, and the first
    # row to claim a point is often not the row that point belongs to
    def setUp(self):
        self.grid = make_grid()

    def tilted_grid(self, angle):
        image = cv2.imread(GRID_FIXTURE)
        edge = np.concatenate(
            [image[0, :], image[-1, :], image[:, 0], image[:, -1]]
        )
        fill = [int(value) for value in np.median(edge, axis=0)]
        padded = cv2.copyMakeBorder(
            image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=fill
        )
        height, width = padded.shape[:2]
        rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        rotated = cv2.warpAffine(
            padded,
            rotation,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill,
        )
        return GridDetector(rotated).detect()

    def center_of(self, item):
        (start_point, end_point) = item.bounds
        return (
            (start_point[0] + end_point[0]) / 2,
            (start_point[1] + end_point[1]) / 2,
        )

    def test_every_well_center_resolves_to_its_own_well(self):
        for row in self.grid.rows:
            for item in row.items:
                self.assertIs(self.grid.find_item(self.center_of(item)), item)

    def first_row_wins(self, grid, point):
        # the lookup this replaced: take the first row whose box contains the point
        # and give up if that row has no well there
        for row in grid.rows:
            if row.contains(point):
                return row.find_item(point)
        return None

    def test_well_centers_still_resolve_on_a_tilted_grid(self):
        grid = self.tilted_grid(5)
        self.assertEqual(grid.dimensions, (8, 12))
        centers = [(item, self.center_of(item))
                   for row in grid.rows for item in row.items]

        # the row boxes overlap by tens of pixels at this angle, so stopping at the
        # first row that claims a point loses a large part of the plate
        lost = [item for item, center in centers
                if self.first_row_wins(grid, center) is not item]
        self.assertGreater(len(lost), 10, "tilted fixture no longer exercises the bug")

        for item, center in centers:
            self.assertIs(grid.find_item(center), item)

    def test_a_point_outside_every_well_resolves_to_nothing(self):
        (start_point, _) = self.grid.rows[0].items[0].bounds
        # up and to the left of the first well, so outside the plate entirely
        self.assertIsNone(self.grid.find_item((start_point[0] - 50, start_point[1] - 50)))


class TestFrameHandlerContourSelection(unittest.TestCase):
    # the background subtractor regularly splits one fly into several blobs.
    # only the largest should count, and the gap between them is not movement
    def setUp(self):
        self.grid = make_grid()
        self.handler = MagicMock()
        self.frame_handler = FrameHandler(
            self.grid, AppSettings(keep_defaults=True), self.handler
        )
        self.frame = np.zeros((411, 640, 3), dtype=np.uint8)

        item = self.grid.rows[3].items[5]
        (start_point, end_point) = item.bounds
        self.item = item
        self.center_x = (start_point[0] + end_point[0]) / 2
        self.center_y = (start_point[1] + end_point[1]) / 2

    def detect_returns(self, contours):
        self.frame_handler.motion_detector.detect = MagicMock(return_value=contours)

    def test_only_the_largest_contour_in_a_well_is_measured(self):
        large = square_contour(self.center_x, self.center_y, 10)
        small = square_contour(self.center_x - 8, self.center_y, 4)

        # the fly sits still, but the frame after is split into two blobs
        self.detect_returns([large])
        self.frame_handler.handle(self.frame, 1)
        self.detect_returns([small, large])
        self.frame_handler.handle(self.frame, 2)

        self.assertEqual(self.handler.handle.call_count, 1)
        event = self.handler.handle.call_args[0][0]
        self.assertEqual(event.item.coords, self.item.coords)
        self.assertEqual(event.point.area, cv2.contourArea(large))
        self.assertEqual(event.last_point.frame_count, 1)
        # the fly never moved, so no distance should be recorded
        self.assertEqual(event.distance, 0)

    def test_contour_order_does_not_change_the_result(self):
        large = square_contour(self.center_x, self.center_y, 10)
        small = square_contour(self.center_x - 8, self.center_y, 4)

        self.detect_returns([large])
        self.frame_handler.handle(self.frame, 1)
        self.detect_returns([large, small])
        self.frame_handler.handle(self.frame, 2)

        self.assertEqual(self.handler.handle.call_count, 1)
        self.assertEqual(self.handler.handle.call_args[0][0].distance, 0)


if __name__ == "__main__":
    unittest.main()
