import math
import functools
from hypothesis import given, assume
from hypothesis.strategies import builds, integers, one_of, none, fractions
import pytest
import numpy as np
import cv2 as cv
from cvtools import Point, Point3, Size


# Hard coded test cases


def make_type_drop_arg(type_, *args):
    if type_ == Point3:
        return type_(*args)
    return type_(*args[:-1])


def test_unary_operators():
    for type_ in (Point, Point3, Size):
        T = functools.partial(make_type_drop_arg, type_)
        assert +T(1, 2, 1) == T(1, 2, 1)
        assert -T(1, 2, -1) == T(-1, -2, 1)
        assert abs(T(-1, -2, 1)) == T(1, 2, 1)
        assert round(T(1.1, 1.91, 2.5)) == T(1, 2, 2)
        assert round(T(1.1, 1.91, 1), ndigits=1) == T(1.1, 1.9, 1.0)
        assert math.floor(T(1.1, 1.9, 1.0)) == T(1, 1, 1)
        assert math.ceil(T(1.1, 1.9, 1.0)) == T(2, 2, 1)


def test_binary_operators():
    for type_ in (Point, Point3, Size):
        T = functools.partial(make_type_drop_arg, type_)
        assert T(1, 3, 0) + T(1, 3, 1) == T(2, 6, 1)
        assert T(1, 3, 1.1) * T(1, 3, 2) == T(1, 9, 2.2)
        assert T(1, 3, 2) - T(2, 3, 0) == T(-1, 0, 2)
        assert T(2, 3, 2) / T(1, 2, 2) == T(2, 1.5, 1)
        assert T(2, 3, 2) // T(1, 2, 2) == T(2, 1, 1)


def test_binary_operators_broadcast():
    for type_ in (Point, Point3, Size):
        T = functools.partial(make_type_drop_arg, type_)
        assert T(1, 3, 0) + 2 == T(3, 5, 2)
        assert T(1, 3, 0) * 2 == T(2, 6, 0)
        assert T(1, 3, 0) - 2 == T(-1, 1, -2)
        assert T(2, 3, 1) / 2 == T(1, 1.5, 0.5)
        assert T(2, 3, 1) // 2 == T(1, 1, 0)


def test_binary_operators_broadcast_rhs():
    for type_ in (Point, Point3, Size):
        T = functools.partial(make_type_drop_arg, type_)
        assert 2 + T(1, 3, 0) == T(3, 5, 2)
        assert 2 * T(1, 3, 0) == T(2, 6, 0)
        assert 2 - T(1, 3, 0) == T(1, -1, 2)
        assert 3 / T(2, 3, 1) == T(1.5, 1, 3)
        assert 2 // T(2, 3, 1) == T(1, 0, 2)


# generated test cases
# using rationals to check math exactly instead of dealing with floating point errors

Rationals = fractions()
PositiveRationals = fractions(min_value=0)
PointStrategy = builds(Point, Rationals, Rationals)
Point3Strategy = builds(Point3, Rationals, Rationals, Rationals)
SizeStrategy = builds(Size, PositiveRationals, PositiveRationals)
ArithmeticTupleStrategy = one_of(PointStrategy, Point3Strategy, SizeStrategy)


@given(ArithmeticTupleStrategy)
def test_unary_operators(o):
    assert +o == o
    assert -o == tuple(-v for v in o)
    assert o == -(-o)
    assert abs(o) == tuple(abs(v) for v in o)


@given(ArithmeticTupleStrategy, one_of(integers(min_value=0, max_value=10), none()))
def test_unary_truncation_operators(o, ndigits):
    assume(all(math.isfinite(v) for v in o))  # cannot round/truncate/etc inf or nan
    assert round(o, ndigits) == tuple(round(v, ndigits) for v in o)
    assert math.floor(o) == tuple(math.floor(v) for v in o)
    assert math.ceil(o) == tuple(math.ceil(v) for v in o)


@given(ArithmeticTupleStrategy, ArithmeticTupleStrategy)
def test_binary_operators(lhs, rhs):
    assume(type(lhs) == type(rhs))

    # commutative
    assert lhs + rhs == rhs + lhs == tuple(l + r for l, r in zip(lhs, rhs))
    assert lhs * rhs == rhs * lhs == tuple(l * r for l, r in zip(lhs, rhs))

    # non-commutative
    assert lhs - rhs == tuple(l - r for l, r in zip(lhs, rhs))
    assume(all(v != 0 for v in rhs))
    assert lhs / rhs == tuple(l / r for l, r in zip(lhs, rhs))
    assert lhs // rhs == tuple(l // r for l, r in zip(lhs, rhs))


@given(ArithmeticTupleStrategy, Rationals)
def test_binary_operators_scalar(lhs, rhs):
    assert lhs + rhs == rhs + lhs == tuple(l + rhs for l in lhs)
    assert lhs * rhs == rhs * lhs == tuple(l * rhs for l in lhs)
    assert lhs - rhs == tuple(l - rhs for l in lhs)
    assert rhs - lhs == tuple(rhs - l for l in lhs)

    if rhs == 0.0:
        with pytest.raises(ZeroDivisionError):
            lhs / rhs

        with pytest.raises(ZeroDivisionError):
            lhs // rhs

    assume(not math.isclose(0, rhs))
    assert lhs / rhs == tuple(l / rhs for l in lhs)
    assert lhs // rhs == tuple(l // rhs for l in lhs)


# OpenCV functions

max_size = 50


def blank_img():
    return np.zeros((max_size, max_size), dtype=np.uint8)


ImagePoint = builds(
    Point,
    integers(min_value=0, max_value=max_size - 1),
    integers(min_value=0, max_value=max_size - 1),
)
EllipseSize = builds(
    Size,
    integers(min_value=1, max_value=max_size),
    integers(min_value=1, max_value=max_size),
)


@given(ImagePoint)
def test_cv_circle(p):
    img = cv.circle(blank_img(), p, 5, 255, -1)
    assert img[p.y, p.x] == 255


@given(ImagePoint)
def test_cv_draw_marker(p):
    img = cv.drawMarker(blank_img(), p, 255)
    assert img[p.y, p.x] == 255


@given(ImagePoint, EllipseSize)
def test_cv_ellipse(p, s):
    img = cv.ellipse(blank_img(), p, s, 0, 0, 360, 255, -1)
    assert img[p.y, p.x] == 255


@given(ImagePoint, ImagePoint)
def test_cv_line(p1, p2):
    img = cv.line(blank_img(), p1, p2, 255, 2)
    assert img[p1.y, p1.x] == 255
    assert img[p2.y, p2.x] == 255
