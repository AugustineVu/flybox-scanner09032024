import cv2
import numpy as np

from custom_types.grid import Grid, Item, Row

# this class is responsible for detecting a grid of items
# the basic algorithm is as follows:
# 1. run a crude circle detection on the image
# 2. get the average radius of the detected circles
# 3. update the average radius and keep running until it converges
# 4. use the converged radius to run a more accurate circle detection
# 5. convert each circle into a rectangular grid item
# 6. sort the items into rows and columns

# ideally, this should have adjustable trackbars to allow for easy tuning,
# but for now you can have fun playing around with the constants below

# doing this greatly improves accuracy, especially in bad lighting conditions
SHOULD_EQUALIZE_HISTOGRAM = True
# cv2 docs recommend using blurring,
# but it seems to make the first detection phase less consistent
# we could play around with the parameters in process_frame()
SHOULD_BLUR_FRAME = False
# apparently powerful, but very slow
# also doesn't work well w/ smaller wells
SHOULD_APPLY_BILATERAL_FILTER = False

# param1: filter out less prominent circles (lower = more circles)
# param2: threshold for circle detection (lower = more circles)
# for the initial detection, we want to be conservative
INITIAL_DETECTION_PARAM_1 = 40
INITIAL_DETECTION_PARAM_2 = 50
# these values should be loose, since the initial detection is very crude
INITIAL_DETECTION_LOWER_BOUND = 0.5
INITIAL_DETECTION_UPPER_BOUND = 1.5

# these are used as a starting point for the first detection phase
# and don't influence the final detected circle count
MAX_INITIAL_CIRCLES = 96 * 2
MIN_INITIAL_CIRCLES = 12
# give up after this many iterations
MAX_ITERATIONS = 100
# point at which we consider the average radius to have converged
# lowering increases accuracy, but increases runtime
# and also increases the chance of getting stuck in a local minimum
CONVERGENCE_THRESHOLD = 0.0005
# if our new average is much larger than the old average,
# we most likely have a bad detection, so we use this to ignore it
FILTER_AVERAGE_THRESHOLD = 1.05

# for our final detection, we're using the estimated average radius
# so we can keep these values tighter
FINAL_DETECTION_LOWER_BOUND = 0.85
FINAL_DETECTION_UPPER_BOUND = 1.15
# same effect as param1 and param2 above, but since we have a better idea of the average radius,
# we can be more aggressive
FINAL_DETECTION_PARAM_1 = 10
FINAL_DETECTION_PARAM_2 = 20

# when converting circles to rectangles,
# we want to account for variations in lighting by using a weighted average
# setting this to 1 will make all circles the same size,
# while setting it to 0 will let all circles keep their original detected size
AVERAGE_RADIUS_ALPHA = 0.75


class GridDetector:
    def __init__(self, frame):
        self.frame = frame
        if SHOULD_EQUALIZE_HISTOGRAM:
            self.clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
        self.processed_frame = None
        self.circles = None
        self.average_radius = None
        self.grid = None

    def process_frame(self):
        # implementation detail: I tried using the average of the first X frames,
        # but it didn't seem to make a difference, even at high values like 1000
        processed_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        if SHOULD_EQUALIZE_HISTOGRAM:
            processed_frame = self.clahe.apply(processed_frame)
        if SHOULD_BLUR_FRAME:
            processed_frame = cv2.medianBlur(processed_frame, 5)
        if SHOULD_APPLY_BILATERAL_FILTER:
            processed_frame = cv2.bilateralFilter(processed_frame, 9, 75, 75)
        return processed_frame

    def get_approximate_average_radius(self):
        max_radius = int(
            min(self.frame.shape[0], self.frame.shape[1]) / MIN_INITIAL_CIRCLES
        )
        min_radius = int(
            min(self.frame.shape[0], self.frame.shape[1]) / MAX_INITIAL_CIRCLES
        )
        min_dist = min_radius * 2
        iterations = 0
        last_average_radius = float("inf")
        while iterations < MAX_ITERATIONS:
            iterations += 1
            detected = cv2.HoughCircles(
                self.processed_frame,
                cv2.HOUGH_GRADIENT,
                1,
                min_dist,
                param1=INITIAL_DETECTION_PARAM_1,
                param2=INITIAL_DETECTION_PARAM_2,
                minRadius=min_radius,
                maxRadius=max_radius,
            )
            if detected is None:
                continue
            average_radius = np.average(detected[0, :, 2])
            if average_radius > last_average_radius * FILTER_AVERAGE_THRESHOLD:
                continue
            if abs(average_radius - last_average_radius) < CONVERGENCE_THRESHOLD:
                break

            min_radius = int(average_radius * INITIAL_DETECTION_LOWER_BOUND)
            max_radius = int(average_radius * INITIAL_DETECTION_UPPER_BOUND)
            min_dist = min_radius * 2
            last_average_radius = average_radius

        if last_average_radius == float("inf"):
            raise Exception("No circles detected")
        return last_average_radius

    def detect_circles(self):
        approximate_average_radius = self.get_approximate_average_radius()
        min_radius = int(approximate_average_radius * FINAL_DETECTION_LOWER_BOUND)
        max_radius = int(approximate_average_radius * FINAL_DETECTION_UPPER_BOUND)
        min_dist = min_radius * 2
        detected = cv2.HoughCircles(
            self.processed_frame,
            cv2.HOUGH_GRADIENT,
            1,
            min_dist,
            param1=FINAL_DETECTION_PARAM_1,
            param2=FINAL_DETECTION_PARAM_2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if detected is None:
            raise Exception("No circles detected")
        circles = detected[0]
        return circles

    def detect(self):
        self.processed_frame = self.process_frame()
        self.circles = self.detect_circles()
        self.average_radius = np.average(self.circles[:, 2])

        grid = [[]]
        for x, y, radius in sorted(self.circles, key=lambda circle: circle[1]):
            radius = (radius * (1 - AVERAGE_RADIUS_ALPHA)) + (
                self.average_radius * AVERAGE_RADIUS_ALPHA
            )
            item = (
                (
                    x - radius,
                    y - radius,
                ),
                (
                    x + radius,
                    y + radius,
                ),
            )
            row = grid[-1]
            # case 1: no items in row yet, so we can't make a comparison
            if len(row) == 0:
                row.append(item)
                continue
            # case 2: item is too far from last item on the y axis, so we're in a new row
            last_item = row[-1]
            distance = item[1][1] - last_item[1][1]
            is_in_new_row = distance > self.average_radius
            if is_in_new_row:
                grid.append([item])
                continue
            # case 3: item is close enough to last item, so we're still in the same row
            row.append(item)

        # sort each row by x coordinate
        for row in grid:
            row.sort(key=lambda item: item[0][0])

        # convert each raw element into a class instance
        total_rows = len(grid)
        for row_index, row in enumerate(grid):
            for col_index, item in enumerate(row):
                # this is the index of each well in the output,
                # meaning we go down each column, then over to the next row
                index = col_index * total_rows + row_index
                row[col_index] = Item(
                    item,
                    index,
                    (row_index, col_index),
                )
            grid[row_index] = Row(row)
        grid = Grid(grid)

        self.grid = grid
        return grid
