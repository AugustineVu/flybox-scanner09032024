import cv2
import numpy as np

from custom_types.grid import Grid
from custom_types.motion import MotionEventHandler
from handlers.frame import MotionEvent

# this class wraps a motion event handler to add debug info
# at the moment, we're using it to show the detected fly and well
# this doesn't really cause as much slowdown as you'd imagine
# and if performance is important, we can hide the video altogether

WELL_COLOR = (0, 255, 0)  # CV2 uses BGR because it hates you
WELL_THICKNESS = 1

FLY_COLOR = (255, 0, 0)
FLY_THICKNESS = 2
LAST_FLY_COLOR = (128, 128, 128)
LAST_FLY_THICKNESS = 1

DISTANCE_COLOR = (0, 0, 255)
DISTANCE_THICKNESS = 1

INDEX_THICKNESS = 0.5
INDEX_COLOR = (255, 255, 255)


class DebugOptions:
    def __init__(self):
        self.draw_index = True
        self.draw_fly = True
        self.draw_distance = True

        self.hidden = False  # hides everything
        self.draw_well = False
        self.print_events = False

    def toggle(self, option):
        setattr(self, option, not getattr(self, option))


class DebugHandler(MotionEventHandler):
    def __init__(self, grid: Grid, handler: MotionEventHandler):
        # MotionEventHandler sets up on_frame, which FrameHandler reads every frame
        super().__init__()
        self.grid = grid
        self.handler = handler
        self.options = DebugOptions()

        # wire this up whatever the index option currently says. draw_indices checks
        # the option itself, so leaving it unset when the option happens to start off
        # only meant two things: FrameHandler blew up looking for on_frame, and
        # turning the option on later had no effect because nothing was listening
        self.overlay = None
        self.on_frame = self.draw_indices

    def draw_indices(self, frame):
        if not self.options.draw_index:
            return

        if self.overlay is None:
            self.overlay = np.zeros(frame.shape, dtype=np.uint8)
            for row in self.grid.rows:
                for item in row.items:
                    cv2.putText(
                        self.overlay,
                        str(item.index + 1),
                        (int(item.bounds[0][0]), int(item.bounds[0][1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        INDEX_THICKNESS,
                        INDEX_COLOR,
                    )

        cv2.bitwise_or(frame, self.overlay, frame)

    def draw_well(self, event: MotionEvent):
        (start, end) = event.point.item.bounds
        (x1, y1) = start
        (x2, y2) = end
        cv2.rectangle(
            event.frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            WELL_COLOR,
            WELL_THICKNESS,
        )

    def draw_fly(self, event: MotionEvent):
        cv2.drawContours(
            event.frame, [event.point.contour], -1, FLY_COLOR, FLY_THICKNESS
        )
        cv2.drawContours(
            event.frame,
            [event.last_point.contour],
            -1,
            LAST_FLY_COLOR,
            LAST_FLY_THICKNESS,
        )

    def draw_distance(self, event: MotionEvent):
        cv2.line(
            event.frame,
            (int(event.point.center[0]), int(event.point.center[1])),
            (int(event.last_point.center[0]), int(event.last_point.center[1])),
            DISTANCE_COLOR,
            DISTANCE_THICKNESS,
        )

    def handle(self, event: MotionEvent):
        self.handler.handle(event)
        if self.options.hidden:
            return

        if self.options.draw_well:
            self.draw_well(event)
        if self.options.draw_fly:
            self.draw_fly(event)
        if self.options.draw_distance:
            self.draw_distance(event)
        if self.options.print_events:
            print(event)
