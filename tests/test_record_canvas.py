import unittest
from unittest.mock import MagicMock

from components.record_canvas import (
    FAILING_COLOR,
    RECOVERED_COLOR,
    RecordCanvas,
)

# RecordCanvas needs a Tk window and a capture device to build, so these drive its
# status methods directly with a stand-in self. that covers the part that matters:
# what the strip says, and that it is refreshed from the main thread


def canvas_with(recorder):
    canvas = MagicMock()
    canvas.recorder = recorder
    canvas.status_text = None
    return canvas


def recorder(failed_writes=0, held=0, error_log="results.errors.log"):
    stub = MagicMock()
    stub.failed_writes = failed_writes
    stub.pending_rows = ["row"] * held
    stub.error_log = error_log
    return stub


class TestRecordingStatusMessage(unittest.TestCase):
    def test_says_nothing_while_saving_works(self):
        (text, color) = RecordCanvas.status_message(canvas_with(recorder()))

        self.assertEqual(text, "")
        self.assertIsNone(color)

    def test_says_nothing_when_not_recording_to_a_file(self):
        # tuning mode has no recorder at all
        (text, _) = RecordCanvas.status_message(canvas_with(None))

        self.assertEqual(text, "")

    def test_warns_while_saves_are_failing(self):
        canvas = canvas_with(recorder(failed_writes=2, held=2))

        (text, color) = RecordCanvas.status_message(canvas)

        self.assertIn("NOT SAVING", text)
        self.assertIn("2 interval(s) held", text)
        self.assertIn("results.errors.log", text)
        self.assertEqual(color, FAILING_COLOR)

    def test_reports_a_past_failure_once_saving_recovers(self):
        # nothing held any more, but the run is no longer clean and should say so
        canvas = canvas_with(recorder(failed_writes=3, held=0))

        (text, color) = RecordCanvas.status_message(canvas)

        self.assertIn("Recovered", text)
        self.assertIn("3 failed save(s)", text)
        self.assertEqual(color, RECOVERED_COLOR)

    def test_shows_only_the_log_file_name_not_the_whole_path(self):
        canvas = canvas_with(
            recorder(failed_writes=1, held=1, error_log="/long/path/to/out.errors.log")
        )

        (text, _) = RecordCanvas.status_message(canvas)

        self.assertIn("out.errors.log", text)
        self.assertNotIn("/long/path", text)


class TestRecordingStatusRefresh(unittest.TestCase):
    def refresh(self, canvas):
        RecordCanvas.refresh_status(canvas)

    def test_does_nothing_without_a_status_label(self):
        canvas = canvas_with(recorder(failed_writes=1, held=1))
        canvas.status_label = None

        self.refresh(canvas)  # must not raise

    def test_writes_the_message_onto_the_label(self):
        canvas = canvas_with(recorder(failed_writes=1, held=1))
        canvas.status_message = lambda: ("something went wrong", FAILING_COLOR)

        self.refresh(canvas)

        canvas.status_label.config.assert_called_once_with(
            text="something went wrong", fg=FAILING_COLOR
        )

    def test_does_not_touch_the_label_when_nothing_changed(self):
        # update runs this every frame, so it has to be cheap when all is well
        canvas = canvas_with(recorder())
        canvas.status_message = lambda: ("steady", None)
        self.refresh(canvas)
        canvas.status_label.config.reset_mock()

        self.refresh(canvas)

        canvas.status_label.config.assert_not_called()

    def test_updates_again_when_the_message_changes(self):
        canvas = canvas_with(recorder())
        canvas.status_message = lambda: ("first", FAILING_COLOR)
        self.refresh(canvas)
        canvas.status_message = lambda: ("second", RECOVERED_COLOR)

        self.refresh(canvas)

        self.assertEqual(canvas.status_label.config.call_count, 2)


class TestRecordCanvasPollsStatus(unittest.TestCase):
    def test_update_refreshes_the_status(self):
        # the recorder flushes on a timer thread, so the strip has to be polled from
        # the main thread rather than pushed to from the thread that fails
        canvas = MagicMock()
        canvas.hidden = False
        canvas.get_frame.return_value = ("frame", 1)

        RecordCanvas.update(canvas)

        canvas.refresh_status.assert_called_once()
