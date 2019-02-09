import cv2


def imshow(img, wait=0, window_name=""):
    if img is None:
        raise ValueError(
            "Image is empty; ensure you are reading from the correct path."
        )
    cv2.imshow(window_name, img)
    return cv2.waitKey(wait) & 0xFF


def imshow_components(labels, *args, **kwargs):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0
    return imshow(labeled_img, *args, **kwargs)


def imshow_autoscale(img, *args, **kwargs):
    scaled = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return imshow(scaled, *args, **kwargs)


def imshow_enlarged(img, scale=10, grid=True, color=200, wait=0, window_name=""):
    h, w = img.shape[:2]
    r = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    if grid:
        r = add_grid(img, scale, color)
    return imshow(r, wait, window_name)


def add_grid(img, scale=10, color=200):

    h, w = img.shape[:2]
    r = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    partial_grid = np.zeros((scale, scale), dtype=bool)
    partial_grid[0, :] = True
    partial_grid[:, 0] = True
    gridlines = np.tile(partial_grid, (h, w))
    r[gridlines] = color

    pad_sizes = ((0, 1), (0, 1), (0, 0)) if len(img.shape) == 3 else ((0, 1), (0, 1))
    r = np.pad(r, pad_sizes, "constant", constant_values)
    r[-1, :] = color
    r[:, -1] = color

    return r
