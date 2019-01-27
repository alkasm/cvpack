import cv2
from pathlib import Path


def imread(imgpath, *args, **kwargs):
    """Reads an image, providing helpful errors on failed reads.

    Allows pathlib.Path objects as well as strings for the imgpath.
    """
    img = cv2.imread(str(imgpath), *args, **kwargs)
    if img is None:
        p = Path(imgpath)
        if p.exists():
            raise ValueError("Image is empty!")
        raise FileNotFoundError(f"Image path {p.absolute()} doesn't exist!")
    return img


def imwrite(imgpath, img, *args, **kwargs):
    """Writes an image, providing helpful errors on failed writes.

    Allows pathlib.Path objects as well as strings for the imgpath.
    Will create the directories included in the imgpath if they don't exist.
    """
    if img is None:
        raise ValueError("Image is empty!")
    Path(imgpath).parent.mkdir(parents=True, exist_ok=True)
    imgpath = str(imgpath)
    return cv2.imwrite(imgpath, img, *args, **kwargs)
