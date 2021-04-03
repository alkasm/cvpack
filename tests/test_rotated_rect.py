from hypothesis import given, assume
from hypothesis.strategies import builds, integers
from cvtools import Rect, RotatedRect, Point
import pytest

Integers = integers(min_value=-1_000_000, max_value=1_000_000)
PositiveIntegers = integers(min_value=0, max_value=1_000_000)

RationalRectStrategy = builds(
    Rect, Integers, Integers, PositiveIntegers, PositiveIntegers,
)


@given(RationalRectStrategy)
def test_rotated_rect_to_from_rect(r):
    def assert_contains(rotated_rect, rect):
        # bounding rect is integral, so check intersection and union
        bound = rotated_rect.bounding_rect()
        assert bound.intersection(rect) == rect.area()
        assert bound.union(rect) == bound.area()

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
        assert r.area() == rr.size.area()
        assert rr.angle % 90 == 0
        assert_contains(rr, r)

    assume(r.height > 1 and r.width > 1)
    for p1, p2, p3 in triplets:
        avgp = (p1 + p3) / 2
        with pytest.raises(ValueError):
            rr = RotatedRect.from_points(p1, avgp, p3)
