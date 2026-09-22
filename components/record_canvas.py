import tkinter as tk
from os import getcwd, path
from tkinter import filedialog
from typing import TYPE_CHECKING

from components.debug_frame import DebugFrame
from components.frame_canvas import FrameCanvas
from components.tuning.motion import TuneMotionFrame
from custom_types.motion import MotionEventHandler
from handlers.debug import DebugHandler
from handlers.file_interval import FileIntervalHandler
from handlers.frame import FrameHandler

if TYPE_CHECKING:
    from components.root_window import RootWindow

# shown in the control strip while recording. a run left alone overnight should not
# need the log file read back to notice that saving has stopped working
FAILING_COLOR = "#b00020"
RECOVERED_COLOR = "#8a6100"


class RecordCanvas(FrameCanvas):
    def __init__(self, window: "RootWindow"):
        super().__init__(window)
        self.hidden = False
        self.filename = None
        self.output_path = None
        # set when recording to a file, which is the only case that can fail to save
        self.recorder = None
        self.status_label = None
        self.status_text = None

        try:
            grid = window.app_state["grid"]
        except KeyError:
            raise Exception("Grid not initialized")

        # handlers
        if window.tuning_mode == "motion":
            handler = MotionEventHandler()
        else:
            self.filename = self.window.settings.get(
                "recording.output_file"
            ) or filedialog.asksaveasfilename(defaultextension=".txt")
            if self.filename == "":
                raise ValueError("No output file selected")
            handler = FileIntervalHandler(
                grid,
                self.filename,
                cleanup_queue=window.cleanup,
                interval=window.settings.get("recording.interval"),
                expected_dimensions=(
                    window.settings.get("grid.rows"),
                    window.settings.get("grid.columns"),
                ),
                record_images=window.app_state.get("record_images"),
            )
            handler.start()
            self.recorder = handler
        # wrap with debug handler to enable visualization
        debug_handler = DebugHandler(grid, handler)
        self.frame_handler = FrameHandler(grid, window.settings, debug_handler)

        # components
        # controls
        self.control_frame = tk.Frame(self.window)
        self.path_label = None
        self.hide_button = None
        self.stop_button = tk.Button(
            self.control_frame, text="Stop", command=self.window.state_manager.idle
        )
        # show these buttons when recording to a file
        if self.filename is not None:
            if self.filename.startswith(getcwd()):
                self.output_path = path.relpath(self.filename)
            else:
                self.output_path = self.filename
            self.path_label = tk.Label(
                self.control_frame, text=f"Output: {self.output_path}"
            )
            self.hide_button = tk.Button(
                self.control_frame, text="Hide", command=self.toggle_hide
            )
            # starts empty, so it takes up no room until there is something to say
            self.status_label = tk.Label(self.control_frame, text="")

        # toggle debug display
        self.debug_frame = DebugFrame(window, debug_handler)

        # show tuning controls based on mode
        self.tuning_frame = None
        if self.window.tuning_mode == "motion":
            self.tuning_frame = TuneMotionFrame(
                window, self.frame_handler.motion_detector
            )

        # dummy frame to hide the canvas
        self.hidden_frame = tk.Frame(self.window)
        tk.Label(self.hidden_frame, text="Hidden").grid()

    def layout(self):
        super().grid()
        self.control_frame.grid(row=1, column=0)
        if self.filename is not None:
            self.path_label.grid(row=0, column=0)
            self.hide_button.grid(row=0, column=1)
            self.status_label.grid(row=1, column=0, columnspan=3)
        self.stop_button.grid(row=0, column=2)
        self.debug_frame.layout(row=2)
        if self.tuning_frame is not None:
            self.tuning_frame.layout(row=3)

    def status_message(self):
        # (text, colour) for the status strip. empty text is the normal case
        if self.recorder is None:
            return ("", None)
        log_name = path.basename(self.recorder.error_log)
        held = len(self.recorder.pending_rows)
        if held:
            return (
                f"NOT SAVING: {held} interval(s) held, retrying. See {log_name}",
                FAILING_COLOR,
            )
        if self.recorder.failed_writes:
            return (
                f"Recovered after {self.recorder.failed_writes} failed save(s). "
                f"See {log_name}",
                RECOVERED_COLOR,
            )
        return ("", None)

    def refresh_status(self):
        # polled from update on the main thread. the recorder flushes from a timer
        # thread, and Tk widgets must not be touched from there
        if self.status_label is None:
            return
        (text, color) = self.status_message()
        if text == self.status_text:
            return
        self.status_text = text
        self.status_label.config(text=text, fg=color or "black")

    def toggle_hide(self):
        if self.hidden:
            self.hidden = False
            self.hide_button.config(text="Hide")
            self.config(width=self.window.width, height=self.window.height)
            self.hidden_frame.grid_forget()
        else:
            self.hidden = True
            self.hide_button.config(text="Show")
            self.config(width=0, height=0)
            # mount at window center
            self.hidden_frame.grid(row=0, column=0)
        self.debug_frame.toggle_hidden()

    def resize_frame(self, frame):
        try:
            border = self.window.app_state["border"]
            (x, y, w, h) = border
            frame = frame[y : y + h, x : x + w]
        except KeyError:
            pass

        return super().resize_frame(frame)

    def update(self):
        frame, frame_count = self.get_frame()
        self.frame_handler.handle(frame, frame_count)
        self.refresh_status()
        if self.hidden:
            self.delete_frame()
        else:
            self.show_frame(frame)
