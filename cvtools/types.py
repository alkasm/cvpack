"""
This module mimics some of the OpenCV built-in types that don't get translated
into Python directly. Generally, these just get mapped to/from tuples.

However, a lot of OpenCV code would benefit from types that describe the tuple,
in addition to named attributes. As OpenCV expects tuples for these datatypes,
subclassing from tuples and namedtuples allows for flexibility without breaking
compatibility.

Wherever it made sense, functionality was copied from OpenCV. For overloaded
CPP functions, there is not a hard rule for deciding which version becomes the
defacto Python method, but in some cases, both have been provided. Alternative
constructors are given in the usual Python way---as classmethods prepended with
the verb "from".

Some liberties have been taken with respect to naming. In particular, camelCase
method names have been translated to snake_case. Some methods and attributes
are provided that OpenCV doesn't contain. 

Aside from Scalars (where you can use a numpy array for vector operations),
the usual arithmetic operations available in OpenCV are available here.

You can average two points by adding them and dividing by two; you can add a
point to a rect to shift it; and so on.
"""

from typing import NamedTuple, NamedTupleMeta
from collections.abc import Sequence
import itertools
import operator
import functools
from math import floor, ceil
import numpy as np
import cv2 as cv


class _NamedTupleMetaBases(NamedTupleMeta):
    def __new__(cls, typename, bases, ns):
        cls_obj = super().__new__(cls, typename + "_nm_base", bases, ns)
        bases = bases + (cls_obj,)
        return type(typename, bases, {})


def _flip_args(f):
    "Create a new function from the original with the arguments reversed"

    @functools.wraps(f)
    def wrapped(*args):
        return f(*args[::-1])

    return wrapped


def _is_sequence(o):
    return isinstance(o, (Sequence, np.ndarray)) and not isinstance(o, str)


class _ArithmeticOperators:
    """Generic mixin for arithmetic operations shared between Point, Size, and Scalar classes."""

    def _unary_op(self, op):
        return type(self)(*map(op, self))

    def _binary_op(self, other, op):
        other_seq = other if _is_sequence(other) else itertools.repeat(other)
        return type(self)(*itertools.starmap(op, zip(self, other_seq)))

    def __add__(self, other):
        return self._binary_op(other, operator.add)

    def __sub__(self, other):
        return self._binary_op(other, operator.sub)

    def __mul__(self, other):
        return self._binary_op(other, operator.mul)

    def __truediv__(self, other):
        return self._binary_op(other, operator.truediv)

    def __floordiv__(self, other):
        return self._binary_op(other, operator.floordiv)

    __radd__ = __add__  # commutative
    __rmul__ = __mul__  # commutative

    def __rsub__(self, other):
        return self._binary_op(other, _flip_args(operator.sub))

    def __rtruediv__(self, other):
        return self._binary_op(other, _flip_args(operator.truediv))

    def __rfloordiv__(self, other):
        return self._binary_op(other, _flip_args(operator.floordiv))

    def __pos__(self):
        return self

    def __neg__(self):
        return self._unary_op(operator.neg)

    def __abs__(self):
        return self._unary_op(abs)

    def __round__(self, ndigits=None):
        return self._unary_op(functools.partial(round, ndigits=ndigits))

    def __floor__(self):
        return self._unary_op(floor)

    def __ceil__(self):
        return self._unary_op(ceil)


class Point(_ArithmeticOperators, metaclass=_NamedTupleMetaBases):
    x: float
    y: float

    def cross(self, point):
        return float(np.cross(self, point))

    def dot(self, point):
        return float(np.dot(self, point))

    def ddot(self, point):
        return self.dot(point)

    def inside(self, rect):
        """checks whether the point is inside the specified rectangle"""
        rect = Rect(*rect)
        return rect.contains(self)


class Point3(_ArithmeticOperators, metaclass=_NamedTupleMetaBases):
    x: float
    y: float
    z: float

    def cross(self, point):
        return type(self)(*np.cross(self, point))

    def dot(self, point):
        return float(np.dot(self, point))

    def ddot(self, point):
        return self.dot(point)


class Size(_ArithmeticOperators, metaclass=_NamedTupleMetaBases):
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
    """Mimics cv::Rect while maintaining compatibility with OpenCV's Python bindings.

    Reference: https://docs.opencv.org/master/d2/d44/classcv_1_1Rect__.html
    """

    x: float
    y: float
    width: float
    height: float

    def tl(self):
        """top left point"""
        return Point(self.x, self.y)

    def br(self):
        """bottom right point"""
        return Point(self.x + self.width, self.y + self.height)

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
        rect ± point (shifting a rectangle by a certain offset)
        rect ± size (expanding or shrinking a rectangle by a certain amount)
        """
        if isinstance(other, Point):
            origin = Point(self.x + other.x, self.y + other.y)
            return self.from_origin(origin, self.size())
        elif isinstance(other, Size):
            size = Size(self.width + other.width, self.height + other.height)
            return self.from_origin(self.tl(), size)
        raise NotImplementedError(
            "Adding to a rectangle generically is ambiguous.\n"
            "Add a Point to shift the top-left point, or a Size to expand the rectangle."
        )

    def __sub__(self, other):
        """Shift or alter the size of the rectangle.
        rect ± point (shifting a rectangle by a certain offset)
        rect ± size (expanding or shrinking a rectangle by a certain amount)
        """
        if isinstance(other, Point):
            origin = Point(self.x - other.x, self.y - other.y)
            return self.from_origin(origin, self.size())
        elif isinstance(other, Size):
            w = max(self.width - other.width, 0)
            h = max(self.height - other.height, 0)
            return self.from_origin(self.tl(), Size(w, h))
        raise NotImplementedError(
            "Subtracting from a rectangle generically is ambiguous.\n"
            "Subtract a Point to shift the top-left point, or a Size to shrink the rectangle."
        )

    def __eq__(self, other):
        return all(a == b for a, b in zip(self, other))

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
        return self

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

    @classmethod
    def from_center(cls, center, size):
        """Alternative constructor using a center point and size."""
        w, h = size
        xc, yc = center
        x = xc - w / 2
        y = yc - h / 2
        return cls(x, y, w, h)

    def slice(self):
        """Returns a slice for a numpy array. Not included in OpenCV.

        img[rect.slice()] == img[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width]
        """
        return slice(self.y, self.y + self.height), slice(self.x, self.x + self.width)

    def center(self):
        """Returns the center of the rectangle as a point (xc, yc). Not included in OpenCV.

        rect.center() == (rect.x + rect.width / 2, rect.y + rect.height / 2)
        """
        return Point(self.x + self.width / 2, self.y + self.height / 2)

    def intersection(self, other):
        """Return the area of the intersection of two rectangles. Not included in OpenCV."""
        other = other if isinstance(other, type(self)) else type(self)(*other)
        if self.empty() or other.empty():
            return 0
        w = min(self.x + self.width, other.x + other.width) - max(self.x, other.x)
        h = min(self.y + self.height, other.y + other.height) - max(self.y, other.y)
        return w * h

    def union(self, other):
        """Return the area of the union of two rectangles. Not included in OpenCV."""
        other = other if isinstance(other, type(self)) else type(self)(*other)
        return self.area() + other.area() - self.intersection(other)


class RotatedRect(NamedTuple):
    center: Point
    size: Size
    angle: float

    def bounding_rect(self):
        """returns the minimal rectangle containing the rotated rectangle"""
        pts = self.points()
        r = Rect.from_points(
            Point(floor(min(pt.x for pt in pts)), floor(min(pt.y for pt in pts))),
            Point(ceil(max(pt.x for pt in pts)), ceil(max(pt.y for pt in pts))),
        )
        return r

    def points(self):
        """returns 4 vertices of the rectangle. The order is bottom left, top left, top right, bottom right."""
        b = np.cos(np.radians(self.angle)) * 0.5
        a = np.sin(np.radians(self.angle)) * 0.5

        pt0 = Point(
            self.center.x - a * self.size.height - b * self.size.width,
            self.center.y + b * self.size.height - a * self.size.width,
        )
        pt1 = Point(
            self.center.x + a * self.size.height - b * self.size.width,
            self.center.y - b * self.size.height - a * self.size.width,
        )

        pt2 = Point(2 * self.center.x - pt0.x, 2 * self.center.y - pt0.y)
        pt3 = Point(2 * self.center.x - pt1.x, 2 * self.center.y - pt1.y)

        return [pt0, pt1, pt2, pt3]

    @classmethod
    def from_points(cls, point1, point2, point3):
        """Any 3 end points of the RotatedRect. They must be given in order (either clockwise or anticlockwise)."""
        point1, point2, point3 = Point(*point1), Point(*point2), Point(*point3)
        center = (point1 + point3) * 0.5
        vecs = [point1 - point2, point2 - point3]
        x = max(np.linalg.norm(pt) for pt in (point1, point2, point3))
        a = min(np.linalg.norm(vecs[0]), np.linalg.norm(vecs[1]))

        # check that given sides are perpendicular
        if abs(vecs[0].dot(vecs[1])) * a > np.finfo(np.float32).eps * 9 * x * (
            np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])
        ):
            raise ValueError(
                "The three points do not define a rotated rect. The three points should form a right triangle."
            )

        # wd_i stores which vector (0,1) or (1,2) will make the width
        # One of them will definitely have slope within -1 to 1
        wd_i = 1 if abs(vecs[1][1]) < abs(vecs[1][0]) else 0
        ht_i = (wd_i + 1) % 2

        angle = np.degrees(np.arctan2(vecs[wd_i][1], vecs[wd_i][0]))
        width = np.linalg.norm(vecs[wd_i])
        height = np.linalg.norm(vecs[ht_i])
        size = Size(width, height)

        return cls(center, size, angle)


class _TermCriteriaType:
    COUNT: int = cv.TermCriteria_COUNT
    MAX_ITER: int = cv.TermCriteria_MAX_ITER
    EPS: int = cv.TermCriteria_EPS


class TermCriteria(_TermCriteriaType, metaclass=_NamedTupleMetaBases):
    type: int = 0
    max_count: int = 0
    epsilon: float = 0

    def is_valid(self):
        is_count = (self.type & self.COUNT) and self.max_count > 0
        is_eps = (self.type & self.EPS) and not np.isnan(self.epsilon)
        return is_count or is_eps
