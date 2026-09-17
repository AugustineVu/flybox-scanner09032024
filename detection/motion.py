import time

import cv2
import numpy as np

from utils.app_settings import AppSettings


class MotionDetector:
    def make_bg_subtractor(self):
        current_time = time.time()
        # only make a new subtractor if we haven't done so recently
        # this improves performance and also helps the subtractor react faster to changes
        if current_time - self.last_init_time < self.bg_reinit_throttle:
            return self.bg_subtractor

        self.last_init_time = current_time
        return cv2.createBackgroundSubtractorKNN(
            history=self.history,
            dist2Threshold=self.dist2_threshold,
            detectShadows=False,
        )

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.last_mean = None
        self.last_init_time = 0

        # keep these hardcoded for now
        self.method = "BG_SUBTRACTOR"
        self.operation = cv2.MORPH_CLOSE

        self.kernel_size = self.settings.get("motion.kernel_size")
        self.blur_size = self.settings.get("motion.blur_size")
        self.history = self.settings.get("motion.history")
        self.dist2_threshold = self.settings.get("motion.dist2_threshold")
        self.diff_threshold = self.settings.get("motion.diff_threshold")
        self.min_contour_area = self.settings.get("motion.min_contour_area")
        self.max_contour_area_fraction = self.settings.get(
            "motion.max_contour_area_fraction"
        )
        self.bg_reinit_threshold = self.settings.get("motion.bg_reinit_threshold")
        self.bg_reinit_throttle = self.settings.get("motion.bg_reinit_throttle")

        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        self.bg_subtractor = self.make_bg_subtractor()

    def get_bg_mask(self, gray):
        mask = self.bg_subtractor.apply(gray)
        mask = cv2.morphologyEx(mask, self.operation, self.kernel, iterations=1)
        if self.blur_size > 0:
            mask = cv2.medianBlur(mask, self.blur_size)
        return mask

    def find_contours(self, frame):
        (contours, _) = cv2.findContours(
            frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # two things in here are never a fly. the subtractor returns a mask covering
        # most of the frame when it reinitialises on a lighting change, and it emits a
        # steady trickle of specks. a speck with zero area doesn't even have a centre:
        # get_contour_center can't take moments of it and falls back to an arbitrary
        # vertex, so whatever distance we measure from it is meaningless
        max_area = self.max_contour_area_fraction * frame.shape[0] * frame.shape[1]
        return [
            contour
            for contour in contours
            if self.min_contour_area <= cv2.contourArea(contour) <= max_area
        ]

    def detect_with_bg_subtractor(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        if abs(mean - (self.last_mean or mean)) > self.bg_reinit_threshold:
            # reinitialize the bg subtractor on lighting changes
            self.bg_subtractor = self.make_bg_subtractor()
            # it's tempting to want to avoid contour detection altogether in this case,
            # but for whatever reason it leads to more false positives once detection resumes
        mask = self.get_bg_mask(gray)
        contours = self.find_contours(mask)
        self.last_mean = mean
        return contours

    def detect_with_diff(self, frame):
        # NOTE: not currently functioning
        # small caveat: we drop the first frame
        # if we use this and want to get really picky about timing, we'll want to wait before recording
        if self.last_frame is None:
            self.last_frame = frame.copy()
            return []

        grayA = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(grayB, grayA)
        _, thresh = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        # we use dilation instead of closing, since we want to expand the area of motion for contour detection
        thresh = cv2.dilate(thresh, iterations=2)

        self.last_frame = frame.copy()
        return self.find_contours(thresh)

    def detect(self, frame):
        if self.method == "BG_SUBTRACTOR":
            return self.detect_with_bg_subtractor(frame)
        elif self.method == "DIFF":
            raise NotImplementedError("DIFF method not currently implemented")
        else:
            raise ValueError(f"Invalid method: {self.method}")

    # debug methods used to tune motion detection
    def update_kernel_size(self, size):
        self.kernel_size = int(size)
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

    def update_iterations(self, iterations):
        self.iterations = int(iterations)

    def update_blur_size(self, size):
        size = int(size)
        # make sure positive values are odd
        if size > 0 and size % 2 == 0:
            size += 1
        self.blur_size = size

    def update_history(self, history):
        self.history = int(history)
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
            history=self.history,
            dist2Threshold=self.dist2_threshold,
            detectShadows=False,
        )

    def update_dist2_threshold(self, dist2_threshold):
        self.dist2_threshold = int(dist2_threshold)
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
            history=self.history,
            dist2Threshold=self.dist2_threshold,
            detectShadows=False,
        )

    def save_settings(self):
        self.settings.set("motion.kernel_size", self.kernel_size)
        self.settings.set("motion.blur_size", self.blur_size)
        self.settings.set("motion.history", self.history)
        self.settings.set("motion.dist2_threshold", self.dist2_threshold)
        self.settings.save()
