import tkinter as tk
from tkinter import messagebox
from typing import TYPE_CHECKING

import cv2

from components.frame_canvas import FrameCanvas
from detection.border import BorderDetector
from detection.grids import GridDetector

if TYPE_CHECKING:
    from components.root_window import RootWindow


class ScanCanvas(FrameCanvas):
    def __init__(self, window: "RootWindow"):
        super().__init__(window)
        self.grid = None
        self.window.app_state.pop("grid", None)
        self.window.app_state.pop("border", None)

        self.hidden = False
        self.border_detector = BorderDetector()

        self.button_frame = tk.Frame()
        self.rescan_button = tk.Button(
            self.button_frame, text="Rescan", command=self.detect_grid
        )
        self.record_button = tk.Button(
            self.button_frame, text="Record", command=self.start_recording
        )
        # start disabled
        self.record_button.config(state=tk.DISABLED)
        self.cancel_button = tk.Button(
            self.button_frame, text="Cancel", command=self.window.state_manager.idle
        )
        self.record_images = tk.BooleanVar()
        self.record_images.set(False)
        self.record_images_checkbox = tk.Checkbutton(
            self.button_frame,
            text="Record images",
            variable=self.record_images,
        )

        self.detect_grid()
        if window.tuning_mode == "motion":
            # need to schedule this to avoid updating a dead canvas.
            # this skips the Record button entirely, so start_recording has to do its
            # own checking rather than trusting that the button was clickable
            self.window.after_idle(self.start_recording)

    def layout(self):
        super().grid()
        self.button_frame.grid(row=1, column=0)
        self.rescan_button.grid(row=0, column=0)
        self.record_button.grid(row=0, column=1)
        self.cancel_button.grid(row=0, column=2)
        self.record_images_checkbox.grid(row=1, column=0, columnspan=3)

    def can_record(self):
        # the single answer to "is this scan good enough to record", used both to set
        # the Record button's state and to check again when recording actually starts
        if self.grid is None:
            return False
        return self.grid.matches_dimensions(
            self.window.settings.get("grid.rows"),
            self.window.settings.get("grid.columns"),
        )

    def start_recording(self):
        if not self.can_record():
            # reachable without the button: --tuning motion calls this directly
            messagebox.showwarning(
                "Cannot Record",
                "No usable grid was detected. Rescan before recording.",
            )
            return
        # only set grid here now that it's confirmed
        self.window.app_state["grid"] = self.grid
        self.window.app_state["record_images"] = self.record_images.get()
        self.window.state_manager.record()

    def draw_grid(self, frame):
        # this is display only. it must not run before detection, or a rescan ends up
        # looking for circles in the rectangles the previous scan drew
        if self.grid is None:
            return

        for row in self.grid.rows:
            for item in row.items:
                (start_point, end_point) = item.bounds
                cv2.rectangle(
                    frame,
                    (int(start_point[0]), int(start_point[1])),
                    (int(end_point[0]), int(end_point[1])),
                    (0, 255, 0),
                    thickness=1,
                )
        cv2.putText(
            frame,
            f"{len(self.grid.rows)}x{len(self.grid.rows[0].items)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            thickness=2,
        )

    def detect_grid(self):
        frame = self.get_frame()[0]
        grid_detector = GridDetector(frame)
        try:
            grid = grid_detector.detect()
        except Exception as e:
            # drop whatever the last scan found: recording against a grid detected
            # from an older frame is worse than refusing to record at all
            self.grid = None
            self.record_button.config(state=tk.DISABLED)
            messagebox.showwarning("Detection Failed", str(e))
            return

        # keep the grid either way, so the overlay shows what was actually found
        self.grid = grid
        expected_rows = self.window.settings.get("grid.rows")
        expected_columns = self.window.settings.get("grid.columns")
        if not self.can_record():
            # either a well was missed, or the wells were grouped into the wrong
            # number of rows. recording either one produces an output file whose
            # columns don't mean what the analysis downstream assumes they mean
            self.record_button.config(state=tk.DISABLED)
            sizes = [len(row.items) for row in grid.rows]
            messagebox.showwarning(
                "Unexpected Grid",
                f"Expected {expected_rows} rows of {expected_columns} wells, "
                f"but found rows of {sizes} wells. "
                "Adjust the lighting or camera and rescan before recording.",
            )
            return
        self.record_button.config(state=tk.NORMAL)

    def resize_frame(self, frame):
        try:
            (x, y, w, h) = self.window.app_state["border"]
        except KeyError:
            (x, y, w, h) = self.border_detector.get_border(frame)
            self.window.app_state["border"] = (x, y, w, h)
        frame = frame[y : y + h, x : x + w]
        return super().resize_frame(frame)

    def update(self):
        # draw the overlay here rather than in get_frame, so that detect_grid gets
        # a frame straight from the camera instead of one we've already drawn on
        frame = self.get_frame()[0]
        self.draw_grid(frame)
        self.show_frame(frame)
