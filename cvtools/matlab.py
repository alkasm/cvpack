import numpy as np
import cv2


def bwdist(
    img,
    method=cv2.DIST_L2,
    dist_mask=cv2.DIST_MASK_5,
    label_type=cv2.DIST_LABEL_CCOMP,
    ravel=True,
):
    """Mimics Matlab's bwdist function, similar to OpenCV's distanceTransform()
    but with different output.

    https://www.mathworks.com/help/images/ref/bwdist.html

    Available metrics:
        https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gaa2bfbebbc5c320526897996aafa1d8eb

    Available distance masks:
        https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gaaa68392323ccf7fad87570e41259b497

    Available label types:
        https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga3fe343d63844c40318ee627bd1c1c42f
    """
    flip = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)[1]
    dist, labeled = cv2.distanceTransformWithLabels(flip, method, dist_mask)

    if ravel:  # return linear indices if ravel == True (default)
        idx = np.zeros(img.shape, dtype=np.intp)
        idx_func = np.flatnonzero
    else:  # return two-channel indices if ravel == False
        idx = np.zeros((*img.shape, 2), dtype=np.intp)
        idx_func = lambda masked: np.dstack(np.where(masked))

    for l in np.unique(labeled):
        mask = labeled == l
        idx[mask] = idx_func(img * mask)
    return dist, idx


def imfill(bin_img):
    """Fills holes in the input binary image.

    Achieves the same output as the imfill(BW, 'holes') variant.

    https://www.mathworks.com/help/images/ref/imfill.html
    """
    contours = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    return cv2.drawContours(bin_img, contours, -1, 255, -1)
