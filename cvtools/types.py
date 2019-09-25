from typing import NamedTuple


class Point(NamedTuple):
    x: float
    y: float

    def cross(self, point):
        raise NotImplementedError

    def dot(self, point):
        raise NotImplementedError

    def ddot(self, point):
        return self.dot(point)

    def inside(self, rect):
        """checks whether the point is inside the specified retangle"""
        return (rect.x <= self.x <= rect.x + rect.width) and (
            rect.y <= self.y <= rect.y + rect.height
        )


class Size(NamedTuple):
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
        return (
            self.x <= point.x <= self.x + self.width
            and self.y <= point.y <= self.y + self.height
        )

    def empty(self):
        """true if empty"""
        return self.width <= 0 or self.height <= 0

    @classmethod
    def from_points(cls, top_left, bottom_right):
        x1, y1 = top_left
        x2, y2 = bottom_right
        w = x2 - x1
        h = y2 - y1
        return cls(x1, y1, w, h)

    @classmethod
    def from_origin(cls, origin, size):
        x, y = origin
        w, h = size
        return cls(x, y, w, h)

    def _slice(rect):
        """Returns a slice for a numpy array. Not included in OpenCV.

        img[rect._slice()] == img[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width]
        """
        return slice(rect.y, rect.y + rect.height), slice(rect.x, rect.x + rect.width)

    def _center(self):
        """Returns the center of the rectangle as a point (xc, yc).

        rect._center() == (rect.x + rect.width / 2, rect.y + rect.height / 2)
        """
        return Point(self.x + self.width / 2, self.y + self.height / 2)
