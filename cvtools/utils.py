import cv2
import numpy as np


def circular_gradient(h, w, center=None, radius=None, invert=False):
    """Makes a gradient, white at the center point, fading out to entirely black
    at radius. Colors are flipped if invert == True.

    By default, center is the center of (h, w) and radius is the distance to
    the nearer edge.
    """
    center = center or (h // 2, w // 2)
    radius = radius or min(h, w) // 2

    mask = np.zeros((h, w), dtype=np.uint8)
    mask = cv2.circle(mask, center=center, radius=radius, color=255, thickness=-1)
    grad = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    grad = grad / grad.max()

    if invert:
        grad = 1 - grad

    return grad


def circular_mask(h, w, center=None, radius=None):

    if center is None:  # use the middle of the image
        center = [int(w / 2), int(h / 2)]
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask
