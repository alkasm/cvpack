from functools import partial
import cvtools
import math
import functools
from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import builds, integers, floats, one_of, none, fractions
import pytest
import numpy as np

# Hard coded test cases


def make_type_drop_arg(type_, *args):
    if type_ == cvtools.Point3:
        return type_(*args)
    return type_(*args[:-1])


def test_unary_operators():
    for type_ in (cvtools.Point, cvtools.Point3, cvtools.Size):
        T = functools.partial(make_type_drop_arg, type_)
        assert +T(1, 2, 1) == T(1, 2, 1)
        assert -T(1, 2, -1) == T(-1, -2, 1)
        assert abs(T(-1, -2, 1)) == T(1, 2, 1)
        assert round(T(1.1, 1.91, 2.5)) == T(1, 2, 2)
        assert round(T(1.1, 1.91, 1), ndigits=1) == T(1.1, 1.9, 1.0)
        assert math.floor(T(1.1, 1.9, 1.0)) == T(1, 1, 1)
        assert math.ceil(T(1.1, 1.9, 1.0)) == T(2, 2, 1)


def test_binary_operators():
    for type_ in (cvtools.Point, cvtools.Point3, cvtools.Size):
        T = functools.partial(make_type_drop_arg, type_)
        assert T(1, 3, 0) + T(1, 3, 1) == T(2, 6, 1)
        assert T(1, 3, 1.1) * T(1, 3, 2) == T(1, 9, 2.2)
        assert T(1, 3, 2) - T(2, 3, 0) == T(-1, 0, 2)
        assert T(2, 3, 2) / T(1, 2, 2) == T(2, 1.5, 1)
        assert T(2, 3, 2) // T(1, 2, 2) == T(2, 1, 1)


def test_binary_operators_broadcast():
    for type_ in (cvtools.Point, cvtools.Point3, cvtools.Size):
        T = functools.partial(make_type_drop_arg, type_)
        assert T(1, 3, 0) + 2 == T(3, 5, 2)
        assert T(1, 3, 0) * 2 == T(2, 6, 0)
        assert T(1, 3, 0) - 2 == T(-1, 1, -2)
        assert T(2, 3, 1) / 2 == T(1, 1.5, 0.5)
        assert T(2, 3, 1) // 2 == T(1, 1, 0)


def test_binary_operators_broadcast_rhs():
    for type_ in (cvtools.Point, cvtools.Point3, cvtools.Size):
        T = functools.partial(make_type_drop_arg, type_)
        assert 2 + T(1, 3, 0) == T(3, 5, 2)
        assert 2 * T(1, 3, 0) == T(2, 6, 0)
        assert 2 - T(1, 3, 0) == T(1, -1, 2)
        assert 3 / T(2, 3, 1) == T(1.5, 1, 3)
        assert 2 // T(2, 3, 1) == T(1, 0, 2)


def test_rect_slice():
    img_rect = cvtools.Rect(0, 0, 160, 90)
    a = np.random.rand(img_rect.height, img_rect.width)
    assert np.all(a == a[img_rect.slice])

    rect = cvtools.Rect(10, 12, 14, 16)
    assert np.all(
        a[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width] == a[rect.slice]
    )

    empty_rect = cvtools.Rect(0, 0, 0, 0)
    assert a[empty_rect.slice].size == 0


# generated test cases
# using rationals to check math exactly instead of dealing with floating point errors

Rationals = fractions()
PositiveRationals = fractions(min_value=0)
PointStrategy = builds(cvtools.Point, Rationals, Rationals)
Point3Strategy = builds(cvtools.Point3, Rationals, Rationals, Rationals)
SizeStrategy = builds(cvtools.Size, PositiveRationals, PositiveRationals)
ArithmeticTupleStrategy = one_of(PointStrategy, Point3Strategy, SizeStrategy)
RectStrategy = builds(
    cvtools.Rect, Rationals, Rationals, PositiveRationals, PositiveRationals
)
Integers100 = integers(min_value=0, max_value=100)
RegionStrategy = builds(
    cvtools.Rect, Integers100, Integers100, Integers100, Integers100
)


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


@given(RectStrategy)
def test_rect(r: cvtools.Rect):
    assert r.area() == r.width * r.height
    assert r.area() == r.size().area()
    assert r.tl() == (r.x, r.y)
    assert r.br() == (r.x + r.width, r.y + r.height)
    assert r.size() == (r.width, r.height)
    assert r.empty() == r.size().empty()
    assert r.empty() == (r.area() == 0)
    assert r.center.x == r.x + r.width / 2
    assert r.center.y == r.y + r.height / 2
    assert r.contains(r.center)
    assert r.contains(r.tl())
    assert r.contains(r.br())

    assert r == cvtools.Rect.from_center(r.center, r.size())
    assert r == cvtools.Rect.from_points(r.tl(), r.br())


@given(RectStrategy)
def test_rect_intersection_union(r):
    assert r.area() == r.intersection(r)
    assert r.area() == r.union(r)

    extended_rect = cvtools.Rect.from_points(r.tl() - 1, r.br() + 1)
    assert r.area() == r.intersection(extended_rect)
    assert extended_rect.area() == r.union(extended_rect)

    non_intersecting_rect = cvtools.Rect.from_origin(r.br(), r.size())
    assert r.intersection(non_intersecting_rect) == 0
    assert r.union(non_intersecting_rect) == r.area() + non_intersecting_rect.area()

    empty_rect = cvtools.Rect(0, 0, 0, 0)
    assert r.intersection(empty_rect) == 0
    assert r.union(empty_rect) == r.area()

    intersecting_rect = cvtools.Rect.from_center(r.br(), r.size())
    assert r.intersection(intersecting_rect) == r.area() / 4
    assert r.union(intersecting_rect) == r.area() * 7 / 4


@given(RectStrategy, PointStrategy)
def test_rect_point_ops(r, p):
    sp = r + p
    assert sp.tl() == r.tl() + p
    assert sp.br() == r.br() + p
    assert sp.center == r.center + p
    assert r.area() == sp.area()
    assert r.size() == sp.size()

    sn = r - p
    assert sn.tl() == r.tl() - p
    assert sn.br() == r.br() - p
    assert sn.center == r.center - p
    assert r.area() == sn.area()
    assert r.size() == sn.size()


@given(RectStrategy, SizeStrategy)
def test_rect_size_ops(r, s):
    e = r + s
    assert e.tl() == r.tl()
    assert e.br() == r.br() + s
    assert e.center == r.center + s / 2
    assert r.intersection(e) == r.area()
    assert r.union(e) == e.area()
    assert e.size() == r.size() + s
    assert r.width <= e.width
    assert r.height <= e.height

    assume(s.width <= r.width and s.height <= r.height)
    c = r - s
    assert c.tl() == r.tl()
    assert c.br() == r.br() - s
    assert c.center == r.center - s / 2
    assert r.intersection(c) == c.area()
    assert r.union(c) == r.area()
    assert c.size() == r.size() - s
    assert c.width <= r.width
    assert c.height <= r.height


@given(RegionStrategy)
def test_rect_slices(r):
    img = np.random.rand(200, 200)
    assert np.all(img[r.y : r.y + r.height, r.x : r.x + r.width] == img[r.slice])
    assert img[r.slice].shape == r.size()[::-1]
    assert img[r.slice].size == r.area()
