from typing import List, Tuple

from custom_types.geometry import Point, Rectangle


class GridComponent:
    # this takes the fly's center rather than its bounding box. a fly resting against a
    # well wall has a bounding box that spills into the gap between wells, which used to
    # match nothing at all. the center is also what distances are measured from,
    # so assignment and measurement now agree on where the fly is
    def contains(self, point: Point):
        try:
            (start_point, end_point) = self.bounds
        except AttributeError:
            raise Exception(f"GridComponent {self} does not have bounds")
        (x, y) = point
        return (
            start_point[0] <= x <= end_point[0]
            and start_point[1] <= y <= end_point[1]
        )


class Item(GridComponent):
    def __init__(self, rectangle: Rectangle, index: int, coords: Tuple[int, int]):
        self.index = index
        self.bounds = rectangle
        self.coords = coords


class Row(GridComponent):
    def __init__(self, items: List[Item]):
        self.items = items
        self.bounds = self.calculate_bounds()

    def __len__(self):
        return len(self.items)

    def calculate_bounds(self):
        if len(self.items) == 0:
            return ((0, 0), (0, 0))
        start_point = (float("inf"), float("inf"))
        end_point = (float("-inf"), float("-inf"))
        for item in self.items:
            (x, y) = item.bounds[0]
            if x < start_point[0]:
                start_point = (x, start_point[1])
            if y < start_point[1]:
                start_point = (start_point[0], y)
            (x, y) = item.bounds[1]
            if x > end_point[0]:
                end_point = (x, end_point[1])
            if y > end_point[1]:
                end_point = (end_point[0], y)
        return (start_point, end_point)

    def find_item(self, point: Point):
        for item in self.items:
            if item.contains(point):
                return item
        return None


class Grid(GridComponent):
    def __init__(self, rows: List[Row]):
        self.rows = rows
        self.bounds = self.calculate_bounds()

    def calculate_bounds(self):
        if len(self.rows) == 0:
            return ((0, 0), (0, 0))
        start_point = (float("inf"), float("inf"))
        end_point = (float("-inf"), float("-inf"))
        for row in self.rows:
            (x, y) = row.bounds[0]
            if x < start_point[0]:
                start_point = (x, start_point[1])
            if y < start_point[1]:
                start_point = (start_point[0], y)
            (x, y) = row.bounds[1]
            if x > end_point[0]:
                end_point = (x, end_point[1])
            if y > end_point[1]:
                end_point = (end_point[0], y)
        return (start_point, end_point)

    def find_row(self, point: Point):
        for row in self.rows:
            if row.contains(point):
                return row
        return None

    def find_item(self, point: Point):
        # a row's bounds are the union of its wells, so on a grid that isn't perfectly
        # level those boxes overlap: the left end of one row sits lower than the right
        # end of the row above. that means the first row to claim a point often isn't
        # the row the point belongs to, so we keep looking instead of giving up
        for row in self.rows:
            if not row.contains(point):
                continue
            item = row.find_item(point)
            if item is not None:
                return item
        return None

    @property
    def dimensions(self):
        return (len(self.rows), len(self.rows[0].items))
