from itertools import tee


def pairwise(iterable):
    """s -> (s0, s1), (s1, s2), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class LineIterator:
    """Iterates through the pixels in a line between two points.

    The line will be clipped on the image boundaries
    The line can be 4- or 8-connected.
    If left_to_right=True, then the iteration is always
    done from the left-most point to the right most,
    not to depend on the ordering of pt1 and pt2 parameters.
    """

    def __init__(self, img, pt1, pt2, connectivity=8, left_to_right=False):

        count = -1
        cols, rows = img.shape[:2]
        self.step = cols
        x1, y1 = pt1
        x2, y2 = pt2

        if not (0, 0, 0, 0) <= (x1, x2, y1, y2) < (cols, cols, rows, rows):
            clip, pt1, pt2 = cv2.clipLine((0, 0, cols, rows), pt1, pt2)
            x1, y1 = pt1
            x2, y2 = pt2
            if not clip:
                self.count = 0
                return

        bt_pix = 1
        istep = self.step

        dx = x2 - x1
        dy = y2 - y1
        s = -1 if dx < 0 else 0

        if left_to_right:
            dx = (dx ^ s) - s
            dy = (dy ^ s) - s
            x1 ^= (x1 ^ x2) & s
            y1 ^= (y1 ^ y2) & s
        else:
            dx = (dx ^ s) - s
            bt_pix = (bt_pix ^ s) - s

        self.index = y1 * istep + x1

        s = -1 if dy < 0 else 0
        dy = (dy ^ s) - s
        istep = (istep ^ s) - s

        # conditional swaps
        s = -1 if dy > dx else 0
        dx ^= dy & s
        dy ^= dx & s
        dx ^= dy & s

        bt_pix ^= istep & s
        istep ^= bt_pix & s
        bt_pix ^= istep & s

        assert dx >= 0 and dy >= 0
        if connectivity == 8:
            self.err = dx - (dy + dy)
            self.plus_delta = dx + dx
            self.plus_step = int(istep)
            self.count = dx + 1
        else:
            self.err = 0
            self.plus_delta = (dx + dx) + (dy + dy)
            self.plus_step = int(istep - bt_pix)
            self.count = dx + dy + 1
        self.minus_delta = -(dy + dy)
        self.minus_step = int(bt_pix)

    def __iter__(self):
        for i in range(self.count):
            y = int(self.index / self.step)
            x = int(self.index - y * self.step)
            yield (x, y)
            mask = -1 if self.err < 0 else 0
            self.err += self.minus_delta + (self.plus_delta & mask)
            self.index += self.minus_step + (self.plus_step & mask)
