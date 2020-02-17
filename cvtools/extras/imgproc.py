import cv2 as cv
import numpy as np


def resize_pad(img, size, pad_color=(0, 0, 0), interpolation=None):

    h, w = img.shape[:2]
    sw, sh = size
    aspect = w / h

    if interpolation is None:
        interpolation = cv.INTER_AREA if (h > sh) or (w > sw) else cv.INTER_CUBIC

    # compute scaling and pad sizing
    if aspect > 1:
        nh = int(np.round(sw / aspect))
        new_size = (sw, nh)
        pad_vert = (sh - nh) // 2
        padding = (pad_vert, sh - pad_vert, 0, 0)
    elif aspect < 1:
        nw = int(np.round(sh * aspect))
        new_size = (nw, sh)
        pad_horz = (sw - nw) // 2
        padding = (0, 0, pad_horz, sw - pad_horz)
    else:
        new_size = (sw, sh)
        padding = (0, 0, 0, 0)

    # scale and pad
    scaled_img = cv.resize(img, new_size, interpolation=interpolation)
    scaled_img = cv.copyMakeBorder(
        scaled_img, *padding, borderType=cv.BORDER_CONSTANT, value=pad_color
    )

    return scaled_img


def circular_gradient(w, h, center=None, radius=None, invert=False):
    """Makes a gradient, white at the center point, fading out to entirely black
    at radius. Colors are flipped if invert == True.

    By default, center is the center of (w, h) and radius is the distance to
    the nearer edge.
    """
    center = center or (w // 2, h // 2)
    radius = radius or min(w, h) // 2

    mask = np.zeros((h, w), dtype=np.uint8)
    mask = cv.circle(mask, center=center, radius=radius, color=255, thickness=-1)
    grad = cv.distanceTransform(mask, cv.DIST_L2, 3)
    grad = grad / grad.max()

    if invert:
        grad = 1 - grad

    return grad


def circular_mask(w, h, center=None, radius=None):
    """Creates a circular binary/logical mask.

    Parameters
    ==========
    w, h : int
        Width and height of the mask.
    center : tuple(numeric, numeric) (default: (h/2, w/2))
        The center of the circular mask. 
    radius : tuple(numeric, numeric) (default: nearest image bound to center)
        Radius of the circle extending from the center point. Note that pixels
        that touch the radius will be included. 
    
    Returns
    =======
    mask : np.ndarray
        Boolean array with shape (h, w) where values are True inside the circle,
        False otherwise.

    Notes
    =====
    From this Stack Overflow answer: https://stackoverflow.com/a/44874588/5087436
    """

    if center is None:  # use the middle of the image
        center = [int(w / 2), int(h / 2)]
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    return dist_from_center <= radius
