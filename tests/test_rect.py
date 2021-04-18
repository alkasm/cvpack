from hypothesis import given, assume
from hypothesis.strategies import builds, integers
import numpy as np
import cv2 as cv
from cvmod import Rect
from .test_point_size import Rationals, PositiveRationals, PointStrategy, SizeStrategy


def test_rect_slice():
    img_rect = Rect(0, 0, 160, 90)
    a = np.random.rand(img_rect.height, img_rect.width)
    assert np.all(a == a[img_rect.slice()])

    rect = Rect(10, 12, 14, 16)
    assert np.all(
        a[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width]
        == a[rect.slice()]
    )

    empty_rect = Rect(0, 0, 0, 0)
    assert a[empty_rect.slice()].size == 0


RectStrategy = builds(Rect, Rationals, Rationals, PositiveRationals, PositiveRationals)
Integers100 = integers(min_value=0, max_value=100)
RegionStrategy = builds(Rect, Integers100, Integers100, Integers100, Integers100)


@given(RectStrategy)
def test_rect(r):
    assert r.area() == r.width * r.height
    assert r.area() == r.size().area()
    assert r.tl() == (r.x, r.y)
    assert r.br() == (r.x + r.width, r.y + r.height)
    assert r.size() == (r.width, r.height)
    assert r.empty() == r.size().empty()
    assert r.empty() == (r.area() == 0)
    assert r.center().x == r.x + r.width / 2
    assert r.center().y == r.y + r.height / 2
    assert r.contains(r.center())
    assert r.contains(r.tl())
    assert r.contains(r.br())

    assert r == Rect.from_center(r.center(), r.size())
    assert r == Rect.from_points(r.tl(), r.br())


@given(RectStrategy)
def test_rect_intersection_union(r):
    assert r.area() == r.intersection(r)
    assert r.area() == r.union(r)

    extended_rect = Rect.from_points(r.tl() - 1, r.br() + 1)
    assert r.area() == r.intersection(extended_rect)
    assert extended_rect.area() == r.union(extended_rect)

    non_intersecting_rect = Rect.from_origin(r.br(), r.size())
    assert r.intersection(non_intersecting_rect) == 0
    assert r.union(non_intersecting_rect) == r.area() + non_intersecting_rect.area()

    empty_rect = Rect(0, 0, 0, 0)
    assert r.intersection(empty_rect) == 0
    assert r.union(empty_rect) == r.area()

    intersecting_rect = Rect.from_center(r.br(), r.size())
    assert r.intersection(intersecting_rect) == r.area() / 4
    assert r.union(intersecting_rect) == r.area() * 7 / 4


@given(RectStrategy, PointStrategy)
def test_rect_point_ops(r, p):
    sp = r + p
    assert sp.tl() == r.tl() + p
    assert sp.br() == r.br() + p
    assert sp.center() == r.center() + p
    assert r.area() == sp.area()
    assert r.size() == sp.size()

    sn = r - p
    assert sn.tl() == r.tl() - p
    assert sn.br() == r.br() - p
    assert sn.center() == r.center() - p
    assert r.area() == sn.area()
    assert r.size() == sn.size()


@given(RectStrategy, SizeStrategy)
def test_rect_size_ops(r, s):
    e = r + s
    assert e.tl() == r.tl()
    assert e.br() == r.br() + s
    assert e.center() == r.center() + s / 2
    assert r.intersection(e) == r.area()
    assert r.union(e) == e.area()
    assert e.size() == r.size() + s
    assert r.width <= e.width
    assert r.height <= e.height

    assume(s.width <= r.width and s.height <= r.height)
    c = r - s
    assert c.tl() == r.tl()
    assert c.br() == r.br() - s
    assert c.center() == r.center() - s / 2
    assert r.intersection(c) == c.area()
    assert r.union(c) == r.area()
    assert c.size() == r.size() - s
    assert c.width <= r.width
    assert c.height <= r.height


@given(RegionStrategy)
def test_rect_slices(r):
    img = np.random.rand(200, 200)
    assert np.all(img[r.y : r.y + r.height, r.x : r.x + r.width] == img[r.slice()])
    assert img[r.slice()].shape == r.size()[::-1]
    assert img[r.slice()].size == r.area()


# OpenCV functions


@given(RegionStrategy)
def test_cv_rectangle(r):
    blank_img = np.zeros((200, 200), dtype=np.uint8)
    pts_img = cv.rectangle(blank_img, r.tl(), r.br(), color=255, thickness=-1)
    rect_img = cv.rectangle(blank_img, r, color=255, thickness=-1)
    assert np.all(pts_img == rect_img)
    assert np.all(pts_img[r.slice()] == 255)
    assert np.all(rect_img[r.slice()] == 255)
    assert not np.any((pts_img == 0)[r.slice()])
    assert not np.any((rect_img == 0)[r.slice()])
