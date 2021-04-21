# cvpack

OpenCV extensions for more Pythonic interactions.

## Install
    
```sh
pip install cvpack-alkasm
```

## Types

`cvpack` includes types that exist in the main C++ OpenCV codebase, but that aren't included in the Python bindings. They are compatible as arguments to OpenCV functions, and they implement the same interfaces (with some new additions). The types that are included are `Point`, `Point3`, `Rect`, `RotatedRect`, `Size`, `TermCriteria`. They are implemented as namedtuples, and as such are immutable.

```python
import cvpack

img = cvpack.imread("img.png")
p1 = cvpack.Point(50, 50)
p2 = cvpack.Point(100, 100)
rect = cvpack.Rect.from_points(p1, p2)
roi = img[rect.slice()]
roi_size = cvpack.Size.from_image(roi)
assert roi_size == rect.size()
```

The overloaded constructors are available as `from_` classmethods, like `from_points` shown above. They also follow the same operator overloads that OpenCV has: two points summed is a point, adding a point to a rectangle shifts it, you can `&` two rectangles to get the intersection as a new rectangle, and so on.

## Image IO

Wrappers for `imread`, `imwrite`, and `imshow` simplify usage by checking errors and allowing path-like objects for path arguments. Additionally, `cvpack` provides functions to read images from a URL (`imread_url`), display to a browser (`imshow_browser`) for statically serving images while working in an interpreter, and displaying images in a Jupyter notebook (`imshow_jupyter`) as HTML directly rather than the typical `plt.imshow` from `matplotlib`. Some other utilities related to display are also included.

```python
from pathlib import Path
import cvpack

for path in Path("folder").glob("*.png"):
    img = cvpack.imread(path)
    big = cvpack.add_grid(cvpack.enlarge(img))
    cvpack.imshow_browser(img, route=str(path))
```

## Video IO

Working with video requires acquiring and releasing resources, so `cvpack` provides context managers for video readers and writers which wrap the classes from OpenCV. Reading video frames is simplified to iterating over the capture object.

```python
import cv2
import cvpack

with cvpack.VideoCapture("video.mp4") as cap:
    with cvpack.VideoWriter("reversed.mp4", fourcc=int(cap.fourcc), fps=cap.fps) as writer:
        for frame in cap:
            flipped = cv2.flip(frame, 0)
            writer.write(flipped)
```
