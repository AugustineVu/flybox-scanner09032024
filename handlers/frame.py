from typing import Dict

from custom_types.grid import Grid
from custom_types.motion import MotionEvent, MotionEventHandler, MotionPoint
from detection.motion import MotionDetector
from utils.app_settings import AppSettings
from utils.geometry import get_contour_center

# this class handles motion detected in frames and emits motion events
# at the moment, this class only handles a single motion event per grid item per frame,
# so we can only handle one fly per well, but this can be changed in the future
# see the logic in handle_contour for more info


class FrameHandler(MotionEvent):
    def __init__(self, grid: Grid, settings: AppSettings, handler: MotionEventHandler):
        self.grid = grid
        self.motion_detector = MotionDetector(settings)
        self.handler = handler

        # the point carried over from the previous frame, per well
        self.points: Dict[tuple, MotionPoint] = {}
        # the best point seen so far in the frame being processed, per well
        self.frame_points: Dict[tuple, MotionPoint] = {}
        self.average = 0

    def find_item(self, center):
        row = self.grid.find_row(center)
        if row is None:
            return
        return row.find_item(center)

    def handle_contour(self, contour, frame_count: int):
        # note that this does not emit anything: it only picks the contour that
        # represents the fly in each well for this frame. emitting as contours arrive
        # would measure the gap between two blobs of the *same* frame as movement
        item = self.find_item(get_contour_center(contour))
        if item is None:
            return

        point = MotionPoint(contour, item, frame_count)
        coords = point.item.coords
        # if we have multiple points in the same frame, we only want to keep the largest one
        # we'll need to change this if we ever want to capture multiple flies in a single well
        best_point = self.frame_points.get(coords)
        if best_point is None or point.area > best_point.area:
            self.frame_points[coords] = point

    def handle(self, frame, frame_count: int):
        contours = self.motion_detector.detect(frame)
        # this is a copy of the original frame used for image recording
        # don't modify it!
        raw_frame = frame.copy()
        # reset per-frame state, then pick one point per well
        self.frame_points = {}
        for contour in contours:
            self.handle_contour(contour, frame_count)
        # now that the frame is resolved, measure against the previous frame's point
        for coords, point in self.frame_points.items():
            last_point = self.points.get(coords)
            if last_point is not None:
                event = MotionEvent(
                    point=point,
                    last_point=last_point,
                    item=point.item,
                    frame=frame,
                    raw_frame=raw_frame,
                )
                self.handler.handle(event)
            self.points[coords] = point
        # HACK: move this after contour detection so that changes to the frame don't affect detection
        # the fact that the same frame is used for detection *and* display is itself bad,
        # but this works for now
        if self.handler.on_frame:
            self.handler.on_frame(frame)
