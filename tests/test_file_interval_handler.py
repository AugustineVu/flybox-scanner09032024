import datetime
import unittest
from queue import SimpleQueue
from threading import Timer
from unittest.mock import MagicMock, mock_open, patch

from handlers.file_interval import DELIMITER, FileIntervalHandler


class TestFileInterval(unittest.TestCase):
    output_path = "tests/fixtures/test_output.txt"
    mock_grid_x = 3
    mock_grid_y = 3
    interval = 10

    def make_mock_grid(self, sizes=None):
        mock_grid = MagicMock()
        mock_grid.rows = []
        if sizes is not None:
            for i, size in enumerate(sizes):
                mock_row = MagicMock()
                mock_row.items = []
                for j in range(size):
                    mock_item = MagicMock()
                    mock_item.coords = (i, j)
                    mock_row.items.append(mock_item)
                mock_grid.rows.append(mock_row)
            mock_grid.is_rectangular = len({*sizes}) == 1
            mock_grid.dimensions = (len(sizes), sizes[0])
            mock_grid.matches_dimensions = (
                lambda rows, columns: mock_grid.is_rectangular
                and mock_grid.dimensions == (rows, columns)
            )
            return mock_grid
        mock_grid.is_rectangular = True
        mock_grid.matches_dimensions = lambda rows, columns: True
        for i in range(self.mock_grid_x):
            mock_row = MagicMock()
            mock_row.items = []
            for j in range(self.mock_grid_y):
                mock_item = MagicMock()
                mock_item.coords = (i, j)
                mock_row.items.append(mock_item)
            mock_grid.rows.append(mock_row)
        return mock_grid

    def setUp(self):
        self.mock_open = mock_open()
        self.patcher = patch("builtins.open", self.mock_open)
        self.patcher.start()

        self.cleanup_queue = SimpleQueue()
        self.error_queue = SimpleQueue()
        self.grid = self.make_mock_grid()
        self.handler = FileIntervalHandler(
            self.grid,
            self.output_path,
            interval=self.interval,
            expected_dimensions=(self.mock_grid_x, self.mock_grid_y),
            cleanup_queue=self.cleanup_queue,
            error_queue=self.error_queue,
        )

        # replace with mock to be safe,
        # since real timers are potentially nasty
        self.real_start = self.handler.start
        self.handler.start = MagicMock()

    def tearDown(self):
        self.patcher.stop()
        self.mock_open.reset_mock()

    def test_initial_state(self):
        # should set default values
        self.assertEqual(self.handler.timer, None)
        self.assertEqual(self.handler.filename, self.output_path)
        self.assertEqual(self.handler.interval, self.interval)
        self.assertEqual(self.handler.index, 0)
        self.assertLessEqual(self.handler.last_flush, datetime.datetime.now())
        self.assertEqual(self.cleanup_queue.qsize(), 1)

        # should set distances from grid dimensions, initialized to 0
        expected_distances = {
            (0, 0): 0,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 0,
            (2, 0): 0,
            (2, 1): 0,
            (0, 2): 0,
            (1, 2): 0,
            (2, 2): 0,
        }
        self.assertEqual(self.handler.distances, expected_distances)
        self.assertEqual(self.handler.max_x, 2)
        self.assertEqual(self.handler.max_y, 2)

        self.mock_open.assert_called_once_with(self.output_path, "w")
        self.mock_open().write.assert_called_once_with("")

    def test_handle(self):
        # should update distance for item at event coords
        event = MagicMock()
        event.item.coords = (0, 0)
        event.distance = 10

        self.handler.handle(event)

        self.assertEqual(self.handler.distances[(0, 0)], 10)

    def test_make_row(self):
        # should return a row string of metadata followed by distances
        self.handler.index = 1
        self.handler.last_flush = datetime.datetime(2022, 1, 1, 0, 0, 0)
        index = 1
        for i in range(3):
            for j in range(3):
                self.handler.distances[(i, j)] = index
                index += 1
        # distances should now look like this:
        # 1 2 3
        # 4 5 6
        # 7 8 9
        expected_row_parts = [1, "01 Jan 22", "00:00:00", 1, 1, 0, 0, "Ct", 0, 0]
        # go down each column, then move to the next row
        expected_distances = [
            1,
            4,
            7,
            2,
            5,
            8,
            3,
            6,
            9,
        ]
        expected_row_parts += expected_distances
        expected_row = DELIMITER.join(map(str, expected_row_parts))

        row = self.handler.make_row()

        self.assertEqual(row, expected_row)

    def test_write_data(self):
        # reset mock, since it's called once in the constructor
        self.mock_open.reset_mock()
        self.handler.make_row = MagicMock()
        self.handler.make_row.return_value = "test_row"

        self.handler.write_data()

        self.mock_open.assert_called_once_with(self.output_path, "a")
        self.mock_open().write.assert_called_once_with("test_row\n")

    def test_flush(self):
        self.handler.index = 1
        self.handler.distances[(0, 0)] = 100
        self.handler.write_data = MagicMock()
        self.handler.start = MagicMock()

        self.handler.flush()

        self.assertEqual(self.handler.index, 2)
        self.assertLessEqual(self.handler.last_flush, datetime.datetime.now())
        self.handler.write_data.assert_called_once()
        self.handler.start.assert_called_once()
        # should be reset back to 0
        self.assertEqual(self.handler.distances[(0, 0)], 0)

    def test_flush_error(self):
        self.handler.write_data = MagicMock(side_effect=Exception)
        self.handler.start = MagicMock()

        self.handler.flush()
        self.handler.flush()
        self.handler.flush()

        self.assertEqual(self.error_queue.qsize(), 3)

    # do *not* use real timers unless you hate yourself
    @patch.object(Timer, "start")
    @patch.object(Timer, "__init__", return_value=None)
    def test_start(self, Timer_init, Timer_start):
        self.real_start()

        Timer_init.assert_called_once_with(self.handler.interval, self.handler.flush)
        Timer_start.assert_called_once()

    def test_refuses_a_ragged_grid(self):
        # a missed well has no honest place in the output, so we refuse the recording
        # rather than write that well out as though the fly in it never moved
        ragged = self.make_mock_grid(sizes=[3, 3, 2])

        with self.assertRaises(ValueError) as caught:
            FileIntervalHandler(
                ragged,
                self.output_path,
                interval=self.interval,
                expected_dimensions=(3, 3),
            )

        self.assertIn("[3, 3, 2]", str(caught.exception))

    def test_refuses_a_uniform_grid_of_the_wrong_shape(self):
        # a tilted grid read as one long row is perfectly rectangular, but every
        # column in the output would refer to the wrong well
        one_long_row = self.make_mock_grid(sizes=[9])
        self.assertTrue(one_long_row.is_rectangular)

        with self.assertRaises(ValueError) as caught:
            FileIntervalHandler(
                one_long_row,
                self.output_path,
                interval=self.interval,
                expected_dimensions=(3, 3),
            )

        self.assertIn("3x3", str(caught.exception))

    def test_refusing_a_bad_grid_leaves_the_output_file_alone(self):
        # the constructor truncates the output file, so it matters that we bail first
        self.mock_open.reset_mock()
        ragged = self.make_mock_grid(sizes=[3, 3, 2])

        with self.assertRaises(ValueError):
            FileIntervalHandler(
                ragged,
                self.output_path,
                interval=self.interval,
                expected_dimensions=(3, 3),
            )

        self.mock_open.assert_not_called()

    def test_interval_is_floored_at_one_second(self):
        # Timer(0, ...) fires immediately, so a sub-second interval used to turn the
        # flush into a tight loop rather than a slow one
        for given in (0, 0.3, "0.5", -5):
            with self.subTest(interval=given):
                handler = FileIntervalHandler(
                    self.make_mock_grid(),
                    self.output_path,
                    interval=given,
                    expected_dimensions=(3, 3),
                )
                self.assertEqual(handler.interval, 1)

    def test_a_normal_interval_is_left_alone(self):
        handler = FileIntervalHandler(
            self.make_mock_grid(),
            self.output_path,
            interval=60,
            expected_dimensions=(3, 3),
        )

        self.assertEqual(handler.interval, 60)

    def test_keeps_recording_after_a_failed_write(self):
        # a write that fails used to skip the reschedule, which quietly ended the run
        self.handler.write_data = MagicMock(side_effect=Exception("disk on fire"))
        self.handler.start = MagicMock()

        self.handler.flush()

        self.handler.start.assert_called_once()
        self.assertEqual(self.error_queue.qsize(), 1)

    def test_recovers_once_writing_works_again(self):
        self.handler.start = MagicMock()
        self.handler.write_data = MagicMock(side_effect=Exception("transient"))
        self.handler.flush()
        self.handler.write_data = MagicMock()

        self.handler.flush()

        self.assertEqual(self.handler.write_data.call_count, 1)
        self.assertEqual(self.handler.start.call_count, 2)

    def test_does_not_reschedule_once_cancelled(self):
        # flush always reschedules now, so cancel has to be the thing that stops it
        self.handler.cancel()

        self.real_start()

        self.assertTrue(self.handler.cancelled)
        self.assertIsNone(self.handler.timer)

    def test_cancel(self):
        self.handler.timer = MagicMock()

        self.handler.cancel()

        self.handler.timer.cancel.assert_called_once()


class TestFileIntervalImages(unittest.TestCase):
    output_path = "tests/fixtures/test_output.txt"

    def make_grid(self):
        grid = MagicMock()
        grid.is_rectangular = True
        grid.matches_dimensions = lambda rows, columns: True
        row = MagicMock()
        row.items = []
        for j in range(2):
            item = MagicMock()
            item.coords = (0, j)
            row.items.append(item)
        grid.rows = [row]
        return grid

    def setUp(self):
        # stop only our own patchers: patch.stopall() would also tear down patchers
        # belonging to other test classes and let them write real files
        for patcher in (patch("builtins.open", mock_open()), patch("os.makedirs")):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.handler = FileIntervalHandler(
            self.make_grid(),
            self.output_path,
            interval=10,
            expected_dimensions=(1, 2),
            record_images=True,
        )
        self.handler.start = MagicMock()

    def test_skips_the_image_when_nothing_moved(self):
        # no motion means handle() never ran, so there is no frame for this interval.
        # write_data used to hand that None straight to cv2.imwrite and blow up
        self.assertIsNone(self.handler.raw_frame)

        with patch("cv2.imwrite") as imwrite:
            self.handler.write_data()

        imwrite.assert_not_called()

    def test_writes_the_image_when_something_moved(self):
        frame = MagicMock()
        self.handler.raw_frame = frame

        with patch("cv2.imwrite") as imwrite:
            self.handler.write_data()

        imwrite.assert_called_once()
        self.assertIs(imwrite.call_args[0][1], frame)

    def test_does_not_reuse_the_previous_intervals_frame(self):
        # writing the last frame we saw under a later timestamp would be a lie
        self.handler.raw_frame = MagicMock()

        with patch("cv2.imwrite") as imwrite:
            self.handler.write_data()
            self.assertEqual(imwrite.call_count, 1)
            self.handler.write_data()
            self.assertEqual(imwrite.call_count, 1)
