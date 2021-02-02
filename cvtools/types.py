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

from typing import NamedTuple, Tuple, Optional, Union
from collections.abc import Sequence
import enum
import numpy as np
import cv2 as cv

SeqOperable = Union[float, Sequence, np.ndarray]


class Point(NamedTuple):
    x: float
    y: float

    def __add__(self, other: SeqOperable) -> "Point":
        return self.__class__(*(np.array(self).__add__(np.array(other))))

    def __sub__(self, other: SeqOperable) -> "Point":
        return self.__class__(*(np.array(self).__sub__(np.array(other))))

    def __mul__(self, other: SeqOperable) -> "Point":
        return self.__class__(*(np.array(self).__mul__(np.array(other))))

    def __truediv__(self, other: SeqOperable) -> "Point":
        return self.__class__(*(np.array(self).__truediv__(np.array(other))))

    def __floordiv__(self, other: SeqOperable) -> "Point":
        return self.__class__(*(np.array(self).__floordiv__(np.array(other))))

    def __rsub__(self, other: SeqOperable) -> "Point":
        return self.__class__(*(np.array(self).__rsub__(np.array(other))))

    def __rtruediv__(self, other: SeqOperable) -> "Point":
        return self.__class__(*(np.array(self).__rtruediv__(np.array(other))))

    def __rfloordiv__(self, other: SeqOperable) -> "Point":
        return self.__class__(*(np.array(self).__rfloordiv__(np.array(other))))

    def __radd__(self, other: SeqOperable) -> "Point":
        return self.__class__(*(np.array(self).__radd__(np.array(other))))

    def __rmul__(self, other: SeqOperable) -> "Point":
        return self.__class__(*(np.array(self).__rmul__(np.array(other))))

    def __pos__(self) -> "Point":
        return self.__class__(*np.array(self).__pos__())

    def __neg__(self) -> "Point":
        return self.__class__(*np.array(self).__neg__())

    def __abs__(self) -> "Point":
        return self.__class__(*np.array(self).__abs__())

    def __round__(self, ndigits: Optional[int] = None) -> "Point":
        return self.__class__(*(round(v, ndigits=ndigits) for v in self))

    def __floor__(self) -> "Point":
        return self.__class__(*np.floor(self))

    def __ceil__(self) -> "Point":
        return self.__class__(*np.ceil(self))

    def cross(self, point: "Point") -> float:
        return float(np.cross(self, point))

    def dot(self, point: "Point") -> float:
        return float(np.dot(self, point))

    def ddot(self, point: "Point") -> float:
        return self.dot(point)

    def inside(self, rect: "Rect") -> bool:
        """checks whether the point is inside the specified rectangle"""
        rect = Rect(*rect)
        return rect.contains(self)


class Point3(NamedTuple):
    x: float
    y: float
    z: float

    def __add__(self, other: SeqOperable) -> "Point3":
        return self.__class__(*(np.array(self).__add__(np.array(other))))

    def __sub__(self, other: SeqOperable) -> "Point3":
        return self.__class__(*(np.array(self).__sub__(np.array(other))))

    def __mul__(self, other: SeqOperable) -> "Point3":
        return self.__class__(*(np.array(self).__mul__(np.array(other))))

    def __truediv__(self, other: SeqOperable) -> "Point3":
        return self.__class__(*(np.array(self).__truediv__(np.array(other))))

    def __floordiv__(self, other: SeqOperable) -> "Point3":
        return self.__class__(*(np.array(self).__floordiv__(np.array(other))))

    def __rsub__(self, other: SeqOperable) -> "Point3":
        return self.__class__(*(np.array(self).__rsub__(np.array(other))))

    def __rtruediv__(self, other: SeqOperable) -> "Point3":
        return self.__class__(*(np.array(self).__rtruediv__(np.array(other))))

    def __rfloordiv__(self, other: SeqOperable) -> "Point3":
        return self.__class__(*(np.array(self).__rfloordiv__(np.array(other))))

    def __radd__(self, other: SeqOperable) -> "Point3":
        return self.__class__(*(np.array(self).__radd__(np.array(other))))

    def __rmul__(self, other: SeqOperable) -> "Point3":
        return self.__class__(*(np.array(self).__rmul__(np.array(other))))

    def __pos__(self) -> "Point3":
        return self.__class__(*np.array(self).__pos__())

    def __neg__(self) -> "Point3":
        return self.__class__(*np.array(self).__neg__())

    def __abs__(self) -> "Point3":
        return self.__class__(*np.array(self).__abs__())

    def __round__(self, ndigits: Optional[int] = None) -> "Point3":
        return self.__class__(*(round(v, ndigits=ndigits) for v in self))

    def __floor__(self) -> "Point3":
        return self.__class__(*np.floor(self))

    def __ceil__(self) -> "Point3":
        return self.__class__(*np.ceil(self))

    def cross(self, point: "Point3") -> "Point3":
        return self.__class__(*np.cross(self, point))

    def dot(self, point: "Point3") -> float:
        return float(np.dot(self, point))

    def ddot(self, point: "Point3") -> float:
        return self.dot(point)


class Size(NamedTuple):
    width: float
    height: float

    def __add__(self, other: SeqOperable) -> "Size":
        return self.__class__(*(np.array(self).__add__(np.array(other))))

    def __sub__(self, other: SeqOperable) -> "Size":
        return self.__class__(*(np.array(self).__sub__(np.array(other))))

    def __mul__(self, other: SeqOperable) -> "Size":
        return self.__class__(*(np.array(self).__mul__(np.array(other))))

    def __truediv__(self, other: SeqOperable) -> "Size":
        return self.__class__(*(np.array(self).__truediv__(np.array(other))))

    def __floordiv__(self, other: SeqOperable) -> "Size":
        return self.__class__(*(np.array(self).__floordiv__(np.array(other))))

    def __rsub__(self, other: SeqOperable) -> "Size":
        return self.__class__(*(np.array(self).__rsub__(np.array(other))))

    def __rtruediv__(self, other: SeqOperable) -> "Size":
        return self.__class__(*(np.array(self).__rtruediv__(np.array(other))))

    def __rfloordiv__(self, other: SeqOperable) -> "Size":
        return self.__class__(*(np.array(self).__rfloordiv__(np.array(other))))

    def __radd__(self, other: SeqOperable) -> "Size":
        return self.__class__(*(np.array(self).__radd__(np.array(other))))

    def __rmul__(self, other: SeqOperable) -> "Size":
        return self.__class__(*(np.array(self).__rmul__(np.array(other))))

    def __pos__(self) -> "Size":
        return self.__class__(*np.array(self).__pos__())

    def __neg__(self) -> "Size":
        return self.__class__(*np.array(self).__neg__())

    def __abs__(self) -> "Size":
        return self.__class__(*np.array(self).__abs__())

    def __round__(self, ndigits: Optional[int] = None) -> "Size":
        return self.__class__(*(round(v, ndigits=ndigits) for v in self))

    def __floor__(self) -> "Size":
        return self.__class__(*np.floor(self))

    def __ceil__(self) -> "Size":
        return self.__class__(*np.ceil(self))

    def area(self) -> float:
        return self.height * self.width

    def empty(self) -> bool:
        """true if empty"""
        return self.width <= 0 or self.height <= 0

    @classmethod
    def from_image(cls, image: np.ndarray) -> "Size":
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

    def __add__(self, other: Union[Point, Size]) -> "Rect":  # type: ignore[override]
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

    def __sub__(self, other: Union[Point, Size]) -> "Rect":
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

    def __and__(self, other: "Rect") -> "Rect":
        """rectangle intersection"""
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        w = min(self.x + self.width, other.x + other.width) - x
        h = min(self.y + self.height, other.y + other.height) - y

        return (
            self.__class__(0, 0, 0, 0)
            if (w <= 0 or h <= 0)
            else self.__class__(x, y, w, h)
        )

    def __or__(self, other: "Rect") -> "Rect":
        """minimum area rectangle containing self and other."""
        if self.empty():
            return other
        elif not other.empty():
            x = min(self.x, other.x)
            y = min(self.y, other.y)
            w = max(self.x + self.width, other.x + other.width) - x
            h = max(self.y + self.height, other.y + other.height) - y
            return self.__class__(x, y, w, h)
        return self

    def tl(self) -> Point:
        """top left point"""
        return Point(self.x, self.y)

    def br(self) -> Point:
        """bottom right point"""
        return Point(self.x + self.width, self.y + self.height)

    def area(self) -> float:
        return self.height * self.width

    def size(self) -> Size:
        """size (width, height) of the rectangle"""
        return Size(self.width, self.height)

    def contains(self, point: Point) -> bool:
        """checks whether the rectangle contains the point"""
        return (
            self.x <= point.x <= self.x + self.width
            and self.y <= point.y <= self.y + self.height
        )

    def empty(self) -> bool:
        """true if empty"""
        return self.width <= 0 or self.height <= 0

    @classmethod
    def from_points(cls, top_left: Point, bottom_right: Point) -> "Rect":
        """Alternative constructor using two points."""
        x1, y1 = top_left
        x2, y2 = bottom_right
        w = x2 - x1
        h = y2 - y1
        return cls(x1, y1, w, h)

    @classmethod
    def from_origin(cls, origin: Point, size: Size) -> "Rect":
        """Alternative constructor using a point and size."""
        x, y = origin
        w, h = size
        return cls(x, y, w, h)

    @classmethod
    def from_center(cls, center: Point, size: Size) -> "Rect":
        """Alternative constructor using a center point and size."""
        w, h = size
        xc, yc = center
        x = xc - w / 2
        y = yc - h / 2
        return cls(x, y, w, h)

    def slice(self) -> Tuple[slice, slice]:
        """Returns a slice for a numpy array. Not included in OpenCV.

        img[rect.slice()] == img[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width]
        """
        return slice(self.y, self.y + self.height), slice(self.x, self.x + self.width)

    def center(self) -> Point:
        """Returns the center of the rectangle as a point (xc, yc). Not included in OpenCV.

        rect.center() == (rect.x + rect.width / 2, rect.y + rect.height / 2)
        """
        return Point(self.x + self.width / 2, self.y + self.height / 2)

    def intersection(self, other: "Rect") -> float:
        """Return the area of the intersection of two rectangles. Not included in OpenCV."""
        if self.empty() or other.empty():
            return 0
        w = min(self.x + self.width, other.x + other.width) - max(self.x, other.x)
        h = min(self.y + self.height, other.y + other.height) - max(self.y, other.y)
        return w * h

    def union(self, other: "Rect") -> float:
        """Return the area of the union of two rectangles. Not included in OpenCV."""
        return self.area() + other.area() - self.intersection(other)


class RotatedRect(NamedTuple):
    center: Point
    size: Size
    angle: float

    def bounding_rect(self) -> Rect:
        """returns the minimal rectangle containing the rotated rectangle"""
        pts = self.points()
        r = Rect.from_points(
            Point(np.floor(min(pt.x for pt in pts)), np.floor(min(pt.y for pt in pts))),
            Point(np.ceil(max(pt.x for pt in pts)), np.ceil(max(pt.y for pt in pts))),
        )
        return r

    def points(self) -> Tuple[Point, Point, Point, Point]:
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

        return pt0, pt1, pt2, pt3

    @classmethod
    def from_points(cls, point1: Point, point2: Point, point3: Point) -> "RotatedRect":
        """Any 3 end points of the RotatedRect. They must be given in order (either clockwise or anticlockwise)."""
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


class TermCriteria(NamedTuple):
    class Type(enum.IntFlag):  # type: ignore[misc]
        COUNT: int = cv.TermCriteria_COUNT
        MAX_ITER: int = cv.TermCriteria_MAX_ITER
        EPS: int = cv.TermCriteria_EPS

    # Without this, the MAX_ITER alias won't show up in some interpreters
    Type._member_names_ = ["COUNT", "MAX_ITER", "EPS"]  # type: ignore[misc]

    type: int = Type.COUNT
    max_count: int = 0
    epsilon: float = 0

    def is_valid(self) -> bool:
        is_count = bool(self.type & self.Type.COUNT) and self.max_count > 0
        is_eps = bool(self.type & self.Type.EPS) and not np.isnan(self.epsilon)
        return is_count or is_eps
