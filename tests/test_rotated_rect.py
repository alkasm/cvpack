from hypothesis import given, assume
from hypothesis.strategies import builds, integers, floats
import numpy as np
import cv2 as cv
from cvtools import Rect, RotatedRect, Point
import pytest
import math

reasonable_size = 1e12

ReasonableFloats = floats(
    min_value=-reasonable_size,
    max_value=reasonable_size,
    allow_infinity=False,
    allow_nan=False,
)
PositiveReasonableFloats = floats(
    min_value=1 / reasonable_size,
    max_value=reasonable_size,
    allow_infinity=False,
    allow_nan=False,
)
FloatRectStrategy = builds(
    Rect,
    ReasonableFloats,
    ReasonableFloats,
    PositiveReasonableFloats,
    PositiveReasonableFloats,
)


@given(FloatRectStrategy)
def test_rotated_rect_to_from_rect(r):
    def assert_contains(rotated_rect, rect):
        # bounding rect is integral, so check intersection and union
        bound = rotated_rect.bounding_rect()
        assert math.isclose(bound.intersection(rect), rect.area(), abs_tol=1e-7)
        assert math.isclose(bound.union(rect), bound.area(), abs_tol=1e-7)

    rr0 = RotatedRect(r.center(), r.size(), 0)
    rr180 = RotatedRect(r.center(), r.size(), 180)
    assert_contains(rr0, r)
    assert_contains(rr180, r)

    tl = r.tl()
    tr = Point(r.x + r.width, r.y)
    br = r.br()
    bl = Point(r.x, r.y + r.height)

    triplets = [
        (tl, tr, br),
        (tr, br, bl),
        (br, bl, tl),
        (bl, tl, tr),
    ]  # clockwise
    triplets += [t[::-1] for t in triplets]  # counter-clockwise

    for p1, p2, p3 in triplets:
        rr = RotatedRect.from_points(p1, p2, p3)
        assert math.isclose(r.area(), rr.size.area(), abs_tol=1e-7)
        assert math.isclose(rr.angle % 90, 0)
        assert_contains(rr, r)

    assume(r.height > 1e-7 and r.width > 1e-7)
    for p1, p2, p3 in triplets:
        avgp = (p1 + p3) / 2
        with pytest.raises(ValueError):
            rr = RotatedRect.from_points(p1, avgp, p3)
