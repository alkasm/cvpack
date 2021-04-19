# cvmod

OpenCV extensions for more Pythonic interactions.

## Install
    
```sh
pip install cvmod-alkasm
```

## Types

`cvmod` includes types that exist in the main C++ OpenCV codebase, but that aren't included in the Python bindings. They are compatible as arguments to OpenCV functions, and they implement the same interfaces (with some new additions). The types that are included are `Point`, `Point3`, `Rect`, `RotatedRect`, `Size`, `TermCriteria`. They are implemented as namedtuples, and as such are immutable.

```python
import cvmod

img = cvmod.imread("img.png")
p1 = cvmod.Point(50, 50)
p2 = cvmod.Point(100, 100)
rect = cvmod.Rect.from_points(p1, p2)
roi = img[rect.slice()]
```

The overloaded constructors are available as `from_` classmethods, like `from_points` shown above. They also follow the same operator overloads that OpenCV has: two points summed is a point, adding a point to a rectangle shifts it, you can `&` two rectangles to get the intersection as a new rectangle, and so on.

## Image IO

Wrappers for `imread`, `imwrite`, and `imshow` simplify usage by checking errors and allowing path-like objects for path arguments. Additionally, `cvmod` provides functions to read images from a URL (`imread_url`), display to a browser (`imshow_browser`) for statically serving images while working in an interpreter, and displaying images in a Jupyter notebook (`imshow_jupyter`) as HTML directly rather than the typical `plt.imshow` from `matplotlib`. Some other utilities related to display are also included.

```python
from pathlib import Path
import cvmod

for path in Path("folder").glob("*.png"):
    img = cvmod.imread(path)
    big = cvmod.add_grid(cvmod.enlarge(img))
    cvmod.imshow_browser(img, route=str(path))
```

## Video IO

Working with video requires acquiring and releasing resources, so `cvmod` provides context managers for video readers and writers which wrap the classes from OpenCV. Reading video frames is simplified to iterating over the capture object.

```python
import cv2
import cvmod

with cvmod.VideoCapture("video.mp4") as cap:
    with cvmod.VideoWriter("reversed.mp4", fourcc=int(cap.fourcc), fps=cap.fps) as writer:
        for frame in cap:
            flipped = cv2.flip(frame, 0)
            writer.write(flipped)
```
