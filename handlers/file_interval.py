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
# problems during a run go here, next to the output file. the file is only created
# when something actually goes wrong, so its presence is itself the warning sign
ERROR_LOG_SUFFIX = ".errors.log"
LOG_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
# a failed write keeps its row so the next flush can save it. cap how many we hold,
# so a disk that never comes back can't grow this without bound
MAX_PENDING_ROWS = 2880
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
        record_images=False,
    ):
        # MotionEventHandler sets up on_frame, which FrameHandler calls on every
        # frame. this only ever worked because a DebugHandler was always wrapped
        # around this one in the app and supplied it
        super().__init__()
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
        # rows that could not be saved yet, oldest first
        self.pending_rows = []
        self.failed_writes = 0
        if cleanup_queue is not None:
            cleanup_queue.put(self.cancel)
        self.raw_frame = None

        self.distances = self.make_distances()
        self.max_x = max(key[0] for key in self.distances)
        self.max_y = max(key[1] for key in self.distances)

        self.filename = filename
        self.error_log = os.path.splitext(filename)[0] + ERROR_LOG_SUFFIX
        self.record_images = record_images
        if self.record_images:
            self.frames_dir = os.path.join(os.path.dirname(filename), "frames")
            os.makedirs(self.frames_dir, exist_ok=True)
        # TODO: move to schema in app settings
        # floor this at a second. Timer(0, ...) fires immediately, so an interval that
        # rounds down to zero turns the flush into a tight loop that pegs a core and
        # writes thousands of rows. float() first so that a fractional interval from
        # the environment lands on the floor instead of raising
        self.interval = max(1, int(float(interval)))
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
        self.save_pending()

    def save_pending(self):
        # last chance to get held rows onto disk. without this, intervals that were
        # waiting on a retry would disappear when the run ends
        if not self.pending_rows:
            return
        try:
            with open(self.filename, "a") as f:
                f.write("".join(row + "\n" for row in self.pending_rows))
            self.log_problem(f"saved {len(self.pending_rows)} held interval(s) on stop")
            self.pending_rows = []
        except Exception as error:
            self.log_problem(
                f"lost {len(self.pending_rows)} unsaved interval(s) on stop: "
                f"{type(error).__name__}: {error}"
            )

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
                # the constructor rejects ragged grids, so every coordinate in this
                # rectangle exists. index directly, so that if that ever stops holding
                # we hear about it instead of writing a well out as zero activity
                row_parts.append(int(self.distances[coords]))

        return DELIMITER.join(map(str, row_parts))

    def log_problem(self, message):
        stamp = datetime.datetime.now().strftime(LOG_TIME_FORMAT)
        line = f"{stamp}{DELIMITER}{message}"
        print(f"[flybox] {line}")
        try:
            with open(self.error_log, "a") as f:
                f.write(line + "\n")
        except Exception:
            # the log is a convenience. it must never be the thing that ends a run
            pass

    def record_failure(self, error):
        self.failed_writes += 1
        if len(self.pending_rows) > MAX_PENDING_ROWS:
            dropped = len(self.pending_rows) - MAX_PENDING_ROWS
            del self.pending_rows[:dropped]
            self.log_problem(
                f"gave up on {dropped} unsaved interval(s) to keep memory bounded"
            )
        self.log_problem(
            f"could not write interval {self.index}: "
            f"{type(error).__name__}: {error}. "
            f"holding {len(self.pending_rows)} interval(s) for the next attempt"
        )

    def write_data(self):
        # hold the row rather than write it straight out, so a write that fails costs
        # us nothing: the next flush saves this interval along with its own
        self.pending_rows.append(self.make_row())
        held = len(self.pending_rows) - 1
        with open(self.filename, "a") as f:
            f.write("".join(row + "\n" for row in self.pending_rows))
        self.pending_rows = []
        if held:
            self.log_problem(f"writing again, and saved {held} held interval(s)")
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
        except Exception as error:
            # deliberately not fatal. losing one interval is bad, losing the rest of
            # an overnight run because one save failed is far worse, so we note it and
            # carry on. anything that makes the run impossible at all has already
            # raised from the constructor, before recording started
            self.record_failure(error)
        finally:
            # reschedule no matter what. an interval that fails to write costs us that
            # interval, but it used to cost every interval after it as well
            self.start()

        self.distances = self.make_distances()
