import cvtools
import math
import functools
from hypothesis import given, assume, settings, Verbosity
from hypothesis.strategies import (
    builds,
    integers,
    floats,
    tuples,
    one_of,
    none,
    composite,
)

# Test unary operations
# Test binary operations same types
# Test binary operations with singular numeric broadcasting


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


FiniteFloats = floats(allow_nan=False, allow_infinity=False)
PointStrategy = builds(cvtools.Point, FiniteFloats, FiniteFloats)
Point3Strategy = builds(cvtools.Point3, FiniteFloats, FiniteFloats, FiniteFloats)
SizeStrategy = builds(cvtools.Size, FiniteFloats, FiniteFloats)
ArithmeticTupleStrategy = one_of(PointStrategy, Point3Strategy, SizeStrategy)

RectStrategy = builds(cvtools.Rect)


@given(ArithmeticTupleStrategy)
def test_unary_operators(o):
    assert +o == o
    assert -o == tuple(-v for v in o)
    assert abs(o) == tuple(abs(v) for v in o)


@given(ArithmeticTupleStrategy, one_of(integers(), none()))
def test_unary_truncation_operators(o, ndigits):
    assume(all(math.isfinite(v) for v in o))  # cannot round/truncate/etc inf or nan
    assert round(o, ndigits) == tuple(round(v, ndigits) for v in o)
    assert math.floor(o) == tuple(math.floor(v) for v in o)
    assert math.ceil(o) == tuple(math.ceil(v) for v in o)


@given(ArithmeticTupleStrategy, ArithmeticTupleStrategy)
def test_binary_operators(lhs, rhs):
    assume(type(lhs) == type(rhs))
    assert lhs + rhs == rhs + lhs == tuple(l + r for l, r in zip(lhs, rhs))
    assert lhs * rhs == rhs * lhs == tuple(l * r for l, r in zip(lhs, rhs))


@settings(verbosity=Verbosity.debug)
@given(ArithmeticTupleStrategy, FiniteFloats)
def test_binary_operators_scalar(lhs, rhs):
    assert lhs + rhs == rhs + lhs == tuple(l + rhs for l in lhs)
    assert lhs * rhs == rhs * lhs == tuple(l * rhs for l in lhs)
    assert lhs - rhs == tuple(l - rhs for l in lhs)
    assert rhs - lhs == tuple(rhs - l for l in lhs)

    assume(not math.isclose(0, rhs))
    assert lhs / rhs == tuple(l / rhs for l in lhs)
