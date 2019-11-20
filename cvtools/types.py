from typing import NamedTuple
import numpy as np


class _IterOps:
    """Generic mixin for arithmetic operations shared between Point & Size classes."""

    def __add__(self, other):
        return type(self)(a + b for a, b in zip(self, other))

    def __sub__(self, other):
        return type(self)(a - b for a, b in zip(self, other))

    def __mul__(self, other):
        if hasattr(other, "__iter__"):
            return type(self)(a * b for a, b in zip(self, other))
        return type(self)(a * other for a in self)  # scalar multiplication

    def __div__(self, other):
        if hasattr(other, "__iter__"):
            return type(self)(a / b for a, b in zip(self, other))
        return type(self)(a / other for a in self)  # scalar division

    def __eq__(self, other):
        return all(a == b for a, b in zip(self, other))


class Point(NamedTuple, _IterOps):
    x: float
    y: float

    def cross(self, point):
        return float(np.cross(self, point))

    def dot(self, point):
        return float(np.dot(self, point))

    def ddot(self, point):
        return self.dot(point)

    def inside(self, rect):
        """checks whether the point is inside the specified retangle"""
        return (rect.x <= self.x <= rect.x + rect.width) and (
            rect.y <= self.y <= rect.y + rect.height
        )


class Point3(NamedTuple, _IterOps):
    x: float
    y: float
    z: float

    def cross(self, point):
        return type(self)(*np.cross(self, point))

    def dot(self, point):
        return float(np.dot(self, point))

    def ddot(self, point):
        return self.dot(point)


class Size(NamedTuple, _IterOps):
    width: float
    height: float

    def area(self):
        return self.height * self.width

    def empty(self):
        """true if empty"""
        return self.width <= 0 or self.height <= 0

    @classmethod
    def from_image(cls, image):
        h, w = image.shape[:2]
        return cls(w, h)


class Rect(NamedTuple):
    x: float
    y: float
    width: float
    height: float

    def tl(self):
        """top left point"""
        return Size(self.x, self.y)

    def br(self):
        """bottom right point"""
        return Size(self.x + self.width, self.y + self.height)

    def area(self):
        return self.height * self.width

    def size(self):
        """size (width, height) of the rectangle"""
        return Size(self.width, self.height)

    def contains(self, point):
        """checks whether the rectangle contains the point"""
        point = Point(*point)
        return (
            self.x <= point.x <= self.x + self.width
            and self.y <= point.y <= self.y + self.height
        )

    def empty(self):
        """true if empty"""
        return self.width <= 0 or self.height <= 0

    def __add__(self, other):
        """Shift or alter the size of the rectangle.
        𝚛𝚎𝚌𝚝 ± 𝚙𝚘𝚒𝚗𝚝 (shifting a rectangle by a certain offset)
        𝚛𝚎𝚌𝚝 ± 𝚜𝚒𝚣𝚎 (expanding or shrinking a rectangle by a certain amount)
        """
        if isinstance(other, Point):
            origin = Point(self.x + other.x, self.y + other.y)
            return self.from_origin(origin, self.size)
        elif isinstance(other, Size):
            size = Size(self.width + other.width, self.height + other.height)
            return self.from_origin(self.tl(), size)
        raise TypeError(
            "Adding to a rectangle generically is ambiguous.\n"
            "Add a Point to shift the top-left point, or a Size to expand the rectangle."
        )

    def __sub__(self, other):
        """Shift or alter the size of the rectangle.
        𝚛𝚎𝚌𝚝 ± 𝚙𝚘𝚒𝚗𝚝 (shifting a rectangle by a certain offset)
        𝚛𝚎𝚌𝚝 ± 𝚜𝚒𝚣𝚎 (expanding or shrinking a rectangle by a certain amount)
        """
        if isinstance(other, Point):
            origin = Point(self.x - other.x, self.y - other.y)
            return self.from_origin(origin, self.size)
        elif isinstance(other, Size):
            size = Size(self.width - other.width, self.height - other.height)
            return self.from_origin(self.tl(), size)
        raise TypeError(
            "Subtracting from a rectangle generically is ambiguous.\n"
            "Subtract a Point to shift the top-left point, or a Size to shrink the rectangle."
        )

    def __and__(self, other):
        """rectangle intersection"""
        other = type(self)(*other)
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        w = min(self.x + self.width, other.x + other.width) - x
        h = min(self.y + self.height, other.y + other.height) - y

        return type(self)(0, 0, 0, 0) if (w <= 0 or h <= 0) else type(self)(x, y, w, h)

    def __or__(self, other):
        """minimum area rectangle containing self and other."""
        other = type(self)(*other)
        if self.empty():
            return other
        elif not other.empty():
            x = min(self.x, other.x)
            y = min(self.y, other.y)
            w = max(self.x + self.width, other.x + other.width) - x
            h = max(self.y + self.height, other.y + other.height) - y
            return type(self)(x, y, w, h)
        return type(self)(0, 0, 0, 0)

    def __eq__(self, other):
        other = type(self)(*other)
        return all(a == b for a, b in zip(self, other))

    @classmethod
    def from_points(cls, top_left, bottom_right):
        """Alternative constructor using two points."""
        x1, y1 = top_left
        x2, y2 = bottom_right
        w = x2 - x1
        h = y2 - y1
        return cls(x1, y1, w, h)

    @classmethod
    def from_origin(cls, origin, size):
        """Alternative constructor using a point and size."""
        x, y = origin
        w, h = size
        return cls(x, y, w, h)

    def _slice(self):
        """Returns a slice for a numpy array. Not included in OpenCV.

        img[rect._slice()] == img[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width]
        """
        return slice(self.y, self.y + self.height), slice(self.x, self.x + self.width)

    def _center(self):
        """Returns the center of the rectangle as a point (xc, yc). Not included in OpenCV.

        rect._center() == (rect.x + rect.width / 2, rect.y + rect.height / 2)
        """
        return Point(self.x + self.width / 2, self.y + self.height / 2)


def _floor(v):
    return int(round(np.floor(v)))


def _ceil(v):
    return int(round(np.ceil(v)))


class RotatedRect(NamedTuple):
    center: Point
    size: Size
    angle: float

    def bounding_rect(self):
        """returns the minimal rectangle containing the rotated rectangle"""
        pts = self.points()
        r = Rect(
            Point(_floor(min(pt.x for pt in pts)), _floor(min(pt.y for pt in pts))),
            Point(_ceil(max(pt.x for pt in pts)), _ceil(max(pt.y for pt in pts))),
        )
        return r

    def points(self):
        """returns 4 vertices of the rectangle. The order is bottom left, top left, top right, bottom right."""
        b = np.cos(np.radians(self.angle)) * 0.5
        a = np.sin(np.radians(self.angle)) * 0.5

        pt0 = Point(
            center.x - a * self.size.height - b * self.size.width,
            center.y + b * self.size.height - a * self.size.width,
        )
        pt1 = Point(
            center.x + a * self.size.height - b * self.size.width,
            center.y - b * self.size.height - a * self.size.width,
        )

        pt2 = Point(2 * center.x - pt0.x, 2 * center.y - pt0.y)
        pt3 = Point(2 * center.x - pt1.x, 2 * center.y - pt1.y)

        return [pt0, pt1, pt2, pt3]

    @classmethod
    def from_points(cls, point1, point2, point3):
        """Any 3 end points of the RotatedRect. They must be given in order (either clockwise or anticlockwise)."""
        point1, point2, point3 = Point(*point1), Point(*point2), Point(*point3)
        center = 0.5 * (point1 + point3)
        vecs = [Point(point1 - point2), Point(point2 - point3)]
        x = max(np.linalg.norm(pt) for pt in (point1, point2, point3))
        a = min(np.linalg.norm(vecs[0]), np.linalg.norm(vecs[1]))

        # check that given sides are perpendicular
        if abs(vecs[0].dot(vecs[1])) * a <= np.finfo(np.float32).eps * 9 * x * (
            np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])
        ):
            raise ValueError(
                "The three points do not define a rotated rect. The three points should form a right triangle."
            )

        # wd_i stores which vector (0,1) or (1,2) will make the width
        # One of them will definitely have slope within -1 to 1
        wd_i = 1 if abs(vecs[1][1]) < abs(vecs[1][0]) else 0
        ht_i = (wd_i + 1) % 2

        angle = np.degrees(np.atan(vecs[wd_i][1] / vecs[wd_i][0]))
        width = np.linalg.norm(vecs[wd_i])
        height = np.linalg.norm(vecs[ht_i])
        size = Size(width, height)

        return cls(center, size, angle)
