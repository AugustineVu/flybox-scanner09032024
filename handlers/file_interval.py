import datetime
import os
from threading import Timer

import cv2

from custom_types.grid import Grid
from custom_types.motion import MotionEvent, MotionEventHandler

# this class handles motion events and flushes them to the specified file at the specified interval

# output file options
# see the make_row method for more info
DATE_FORMAT = "%d %b %y"
TIME_FORMAT = "%H:%M:00"
DELIMITER = "\t"


# captures motion events and flushes them to the specified file at the specified interval
class FileIntervalHandler(MotionEventHandler):
    def __init__(
        self,
        grid: Grid,
        filename: str,
        interval,
        expected_dimensions,
        cleanup_queue=None,
        error_queue=None,
        record_images=False,
    ):
        self.timer = None
        self.cancelled = False
        if not grid.matches_dimensions(*expected_dimensions):
            # better to fail here than to start a run whose output misreports which
            # well each column belongs to, or reports a well that was never detected
            # as simply having no activity
            expected_rows, expected_columns = expected_dimensions
            raise ValueError(
                f"Expected a {expected_rows}x{expected_columns} grid but rows "
                f"contain {[len(row.items) for row in grid.rows]} wells. "
                "Rescan before recording."
            )
        self.grid = grid
        self.error_queue = error_queue
        if cleanup_queue is not None:
            cleanup_queue.put(self.cancel)
        self.raw_frame = None

        self.distances = self.make_distances()
        self.max_x = max(key[0] for key in self.distances)
        self.max_y = max(key[1] for key in self.distances)

        self.filename = filename
        self.record_images = record_images
        if self.record_images:
            self.frames_dir = os.path.join(os.path.dirname(filename), "frames")
            os.makedirs(self.frames_dir, exist_ok=True)
        # TODO: move to schema in app settings
        self.interval = int(interval)
        self.index = 0
        self.last_flush = datetime.datetime.now()

        # immediately write file (provides better feedback, and helps catch errors)
        with open(self.filename, "w") as f:
            f.write("")

    def start(self):
        # a flush that lands while we're shutting down would otherwise queue up
        # another timer behind cancel() and keep the process alive
        if self.cancelled:
            return
        self.timer = Timer(self.interval, self.flush)
        self.timer.start()

    def cancel(self):
        self.cancelled = True
        if self.timer is not None:
            self.timer.cancel()

    def handle(self, event: MotionEvent):
        self.distances[event.item.coords] += event.distance
        if self.record_images:
            self.raw_frame = event.raw_frame

    def make_distances(self):
        distances = {}
        for row in self.grid.rows:
            for item in row.items:
                distances[item.coords] = 0
        return distances

    def make_row(self):
        row_parts = [
            self.index,
            self.last_flush.strftime(DATE_FORMAT),
            self.last_flush.strftime(TIME_FORMAT),
            1,  # monitor status (always 1)
            1,  # monitor number (should be user-specified in the future)
            0,  # unused
            0,  # unused
            "Ct",  # ??
            0,  # unused
            0,  # unused
        ]

        """
        Basically all downstream analysis 
        is written with the order of flies 
        by column, then row, i.e. 1A-H, 2A-H, etc.
        """
        for y in range(self.max_y + 1):
            for x in range(self.max_x + 1):
                coords = (x, y)
                row_parts.append(int(self.distances[coords]))

        return DELIMITER.join(map(str, row_parts))

    def write_data(self):
        row = self.make_row()
        with open(self.filename, "a") as f:
            f.write(row + "\n")
        if self.record_images:
            self.write_image()

    def write_image(self):
        # nothing moved during this interval, so we never captured a frame for it.
        # skip the image rather than crash, and rather than writing out the previous
        # interval's frame under this interval's timestamp
        if self.raw_frame is None:
            return
        timestamp = self.last_flush.strftime("%Y%m%d%H%M%S")
        cv2.imwrite(os.path.join(self.frames_dir, f"{timestamp}.jpg"), self.raw_frame)
        self.raw_frame = None

    def flush(self):
        self.index += 1
        self.last_flush = datetime.datetime.now()
        try:
            self.write_data()
        except Exception as e:
            # since we're running in a thread, we add them to the queue to be handled on the next loop
            if self.error_queue is not None:
                self.error_queue.put(e)
        finally:
            # reschedule no matter what. an interval that fails to write costs us that
            # interval, but it used to cost every interval after it as well
            self.start()

        self.distances = self.make_distances()
