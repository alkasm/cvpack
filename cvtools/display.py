import cv2

def imshow(img, wait=0, window_name=''):
    if img is None:
        raise ValueError('Image is empty; ensure you are reading from the correct path.')
    cv2.imshow(window_name, img)
    return cv2.waitKey(wait) & 0xFF

def imshow_components(labels, *args, **kwargs):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    return imshow(labeled_img, *args, **kwargs)

def imshow_autoscale(img, *args, **kwargs):
    scaled = cv2.normalize(img, None, np.min(img), np.max(img), cv2.NORM_MINMAX, cv2.CV_8U)
    return imshow(scaled, *args, **kwargs)