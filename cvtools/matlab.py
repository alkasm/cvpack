def bwdist(img, metric=cv2.DIST_L2, dist_mask=cv2.DIST_MASK_5, label_type=cv2.DIST_LABEL_CCOMP, ravel=True):
    """Mimics Matlab's bsdist function.

    Available metrics:
        https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gaa2bfbebbc5c320526897996aafa1d8eb
    Available distance masks:
        https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gaaa68392323ccf7fad87570e41259b497
    Available label types:
        https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga3fe343d63844c40318ee627bd1c1c42f
    """
    flip = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)[1]
    dist, labeled = cv2.distanceTransformWithLabels(flip, metric, dist_mask)

    # return linear indices if ravel == True (default)
    if ravel:  
        idx = np.zeros(img.shape, dtype=np.intp)  # np.intp type is for indices
        for l in np.unique(labeled):
            mask = labeled == l
            idx[mask] = np.flatnonzero(img * mask)
        return dist, idx

    # return two-channel indices if ravel == False
    idx = np.zeros((*img.shape, 2), dtype=np.intp)  
    for l in np.unique(labeled):
        mask = labeled == l
        idx[mask] = np.dstack(np.where(img * mask))
    return dist, idx

