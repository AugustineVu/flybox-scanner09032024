import os
import unittest
from tkinter import TclError
from unittest.mock import MagicMock, patch

import cv2

from components.frame_canvas import FrameCanvas
from components.root_window import RootWindow
from detection.border import BorderDetector
from detection.grids import GridDetector
from handlers.file_interval import DELIMITER, FileIntervalHandler
from handlers.frame import FrameHandler
from utils.app_settings import AppSettings

# This used to drive the whole pipeline through Tk and compare against saved totals,
# which made it useless as a regression guard. Both the flush timer and the background
# subtractor reinit throttle run off the wall clock, so how much motion landed in each
# row depended on how fast the machine happened to be: runs varied by hundreds against
# a tolerance of 100, and the test failed whatever the code did.
#
# It is split in two now. TestPipelineEndToEnd replays the video with the clock and the
# interval boundaries under the control of the test, which makes it reproducible to the
# digit. TestAppWiring watches the Tk plumbing without measuring anything.

SOURCE_FILE = "tests/fixtures/video.mp4"
TOTALS_FILE = "tests/fixtures/totals.txt"
OUTPUT_FILE = "tests/test_output.txt"

# the app advances a frame every window.frame_delay milliseconds. holding the clock to
# that rate keeps the reinit throttle behaving the way it would in a real run
FRAME_DELAY_SECONDS = 0.03
# the video is 10 seconds at about 30fps, giving the 10 intervals the old test used
FRAMES_PER_INTERVAL = 30
EXPECTED_DIMENSIONS = (8, 12)
EXPECTED_ROWS = 10

# the KNN subtractor replaces samples in its model at random, drawing on the global
# OpenCV RNG. that state carries across runs inside one process, so the seed has to be
# pinned or two replays of the same video disagree
RNG_SEED = 0

# flip to True only when a change is meant to move the numbers, then read the diff
SHOULD_UPDATE_TOTALS = False


class FakeClock:
    # stands in for the time module inside the motion detector. starts well above zero
    # because make_bg_subtractor throttles against last_init_time = 0 on its first call
    # and would hand back a subtractor it has not created yet
    def __init__(self, step, start=1_000_000.0):
        self.now = start
        self.step = step

    def tick(self):
        self.now += self.step

    def time(self):
        return self.now


def resize_like_the_app(frame, width, height):
    # the real geometry, borrowed from FrameCanvas. all it needs from a canvas is the
    # window size, and its call to self.resize does nothing on a stand-in
    canvas = MagicMock()
    canvas.window.width = width
    canvas.window.height = height
    return FrameCanvas.resize_frame(canvas, frame)


class TestPipelineEndToEnd(unittest.TestCase):
    # replays the fixture video through the real detection and recording code
    def setUp(self):
        self.settings = AppSettings(keep_defaults=True)
        self.width = self.settings.get("window.width")
        self.height = self.settings.get("window.height")
        self.addCleanup(self.remove_output)

    def remove_output(self):
        for path in (OUTPUT_FILE, os.path.splitext(OUTPUT_FILE)[0] + ".errors.log"):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def prepare(self, frame, border):
        (x, y, w, h) = border
        cropped = frame[y : y + h, x : x + w]
        return resize_like_the_app(cropped, self.width, self.height)

    def replay(self):
        cv2.setRNGSeed(RNG_SEED)
        capture = cv2.VideoCapture(SOURCE_FILE)
        self.addCleanup(capture.release)
        ok, first = capture.read()
        self.assertTrue(ok, "could not read the fixture video")

        border = BorderDetector().get_border(first)
        grid = GridDetector(self.prepare(first, border)).detect()
        self.assertEqual(grid.dimensions, EXPECTED_DIMENSIONS)

        recorder = FileIntervalHandler(
            grid,
            OUTPUT_FILE,
            interval=60,
            expected_dimensions=EXPECTED_DIMENSIONS,
        )
        # the test decides when intervals end, so the timer stays out of it
        recorder.start = lambda: None

        # no rewind: the first frame has already been spent on detection, exactly as
        # the app spends it, and seeking back into an H.264 stream is not guaranteed
        # to decode identically to reading it straight through
        clock = FakeClock(FRAME_DELAY_SECONDS)
        count = 0
        with patch("detection.motion.time", clock):
            # built inside the patch so the detector measures its throttle against the
            # same clock the replay advances
            frame_handler = FrameHandler(grid, self.settings, recorder)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                count += 1
                clock.tick()
                frame_handler.handle(self.prepare(frame, border), count)
                if count % FRAMES_PER_INTERVAL == 0:
                    recorder.flush()
        self.assertEqual(recorder.failed_writes, 0)
        return count

    def read_totals(self):
        with open(OUTPUT_FILE) as f:
            rows = [line for line in f.read().splitlines() if line]
        # the first ten fields are metadata, everything after is one column per well
        return [sum(int(part) for part in row.split(DELIMITER)[10:]) for row in rows]

    def test_pipeline_totals_match_the_saved_run(self):
        self.replay()
        totals = self.read_totals()
        self.assertEqual(len(totals), EXPECTED_ROWS)

        if SHOULD_UPDATE_TOTALS:
            with open(TOTALS_FILE, "w") as f:
                for total in totals:
                    f.write(f"{total}\n")
            self.skipTest("totals regenerated")

        with open(TOTALS_FILE) as f:
            expected = [int(line) for line in f.read().splitlines() if line]
        # no tolerance. a deterministic replay either reproduces the run or it does not
        self.assertEqual(totals, expected[:EXPECTED_ROWS])

    def test_replaying_twice_gives_the_same_answer(self):
        self.replay()
        first = self.read_totals()
        self.remove_output()
        self.replay()

        self.assertEqual(first, self.read_totals())


class TestAppWiring(unittest.TestCase):
    # checks the Tk side wires together, without measuring anything timing dependent
    def setUp(self):
        os.environ["SOURCE"] = SOURCE_FILE
        os.environ["OUTPUT_FILE"] = OUTPUT_FILE
        # long enough that the recorder timer cannot fire during the test
        os.environ["INTERVAL"] = "3600"
        self.root_window = RootWindow(args=MagicMock(silent=True, keep_defaults=True))
        self.addCleanup(self.close)

    def close(self):
        try:
            self.root_window.on_close()
        except TclError:
            pass
        for path in (OUTPUT_FILE, os.path.splitext(OUTPUT_FILE)[0] + ".errors.log"):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def scan(self):
        idle_canvas = self.root_window.children["!idlecanvas"]
        idle_canvas.button_frame.children["!button"].invoke()
        return self.root_window.children["!scancanvas"]

    def test_scanning_then_recording_drives_frames_through_the_handler(self):
        scan_canvas = self.scan()
        self.assertTrue(scan_canvas.can_record())

        scan_canvas.record_button.invoke()
        record_canvas = self.root_window.children["!recordcanvas"]
        self.assertEqual(
            self.root_window.app_state["grid"].dimensions, EXPECTED_DIMENSIONS
        )
        self.assertTrue(os.path.exists(OUTPUT_FILE))

        # pump a fixed number of frames rather than handing control to mainloop
        for _ in range(20):
            record_canvas.update()

        self.assertEqual(record_canvas.frame_count, 20)

    def test_a_bad_scan_cannot_start_recording(self):
        scan_canvas = self.scan()
        scan_canvas.grid = None

        with patch("components.scan_canvas.messagebox") as messagebox:
            scan_canvas.start_recording()

        self.assertNotIn("!recordcanvas", self.root_window.children)
        messagebox.showwarning.assert_called_once()
