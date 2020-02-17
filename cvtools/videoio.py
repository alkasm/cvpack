import cv2 as cv


_VIDEO_CAPTURE_PROPS_LIST = [
    "CAP_PROP_POS_MSEC",
    "CAP_PROP_POS_FRAMES",
    "CAP_PROP_POS_AVI_RATIO",
    "CAP_PROP_FRAME_WIDTH",
    "CAP_PROP_FRAME_HEIGHT",
    "CAP_PROP_FPS",
    "CAP_PROP_FOURCC",
    "CAP_PROP_FRAME_COUNT",
    "CAP_PROP_FORMAT",
]

_VIDEO_CAPTURE_PROPS = {
    prop.split("CAP_PROP_").pop().lower(): prop for prop in _VIDEO_CAPTURE_PROPS_LIST
}


def _add_props(props):
    def deco(cls):
        for prop in props:
            setattr(cls, prop, VideoCaptureProperty(prop))
        return cls

    return deco


class VideoCaptureProperty:
    """Descriptors to alias cap.get(cv.CAP_PROP_*) and cap.set(cv.CAP_PROP_*, value).

    Raises AttributeError when setting if that property is not supported.
    """

    _set_err = (
        "The property {p} is not supported by\n"
        "the backend used by the cv.VideoCapture() instance."
    )

    def __init__(self, name):
        self.name = name
        self.prop = getattr(cv, _VIDEO_CAPTURE_PROPS[name])

    def __get__(self, obj, objtype=None):
        return obj.cap.get(self.prop)

    def __set__(self, obj, value):
        if not obj.cap.set(self.prop, value):
            raise AttributeError(self._set_err.format(p=_VIDEO_CAPTURE_PROPS[self.name]))


@_add_props(_VIDEO_CAPTURE_PROPS.keys())
class VideoCapture:
    """An adapter for `cv.VideoCapture`, giving a more Pythonic interface."""

    def __init__(self, *args, **kwargs):
        self.cap = cv.VideoCapture(*args, **kwargs)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video source:", *args, **kwargs)

    def __getattr__(self, key):
        return getattr(self.cap, key)

    def __iter__(self):
        """Resets the frame position to 0 at the start for repeatable iteration."""
        noread = (False, None)
        if self.cap.isOpened():
            for _, frame in iter(self.cap.read, noread):
                yield frame

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exctype, exc, exctrace):
        """Releases the video capture object on exiting the context manager."""
        self.cap.release()


class VideoWriter:

    _nowriter = object()

    def __init__(self, filename, fourcc="mp4v", fps=30, frameSize=None, **kwargs):
        self.filename = str(filename)
        self.fourcc = (
            fourcc if isinstance(fourcc, int) else cv.VideoWriter_fourcc(*fourcc)
        )
        self.fps = fps
        self.kwargs = kwargs

        # wait to create writer based on first frame if size is not provided
        self.writer = (
            self._nowriter if frameSize is None else self._makewriter(frameSize)
        )

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.release()

    def _makewriter(self, frame_size):
        return cv.VideoWriter(
            filename=self.filename,
            fourcc=self.fourcc,
            fps=self.fps,
            frameSize=frame_size,
            **self.kwargs
        )

    def write(self, frame):
        try:
            return self.writer.write(frame)
        except AttributeError as e:
            if self.writer is self._nowriter:
                h, w = frame.shape[:2]
                self.writer = self._makewriter((w, h))
                self.write(frame)
            else:
                raise e

    def release(self):
        try:
            self.writer.release()
        except AttributeError as e:
            if self.writer is not self._nowriter:
                raise e


class VideoPlayer:

    _actions = {"quit": set(map(ord, "\x1bq"))}

    def __init__(self, cap):
        self.cap = cap
        self.rate = int(1000 / self.cap.fps)

    def play(self, window_name="VideoPlayer", framefunc=None, loop=False):
        """Plays through the video file with OpenCV's imshow()."""

        framefunc = framefunc or (lambda frame: frame)

        for frame in self.cap:
            cv.imshow(window_name, framefunc(frame))
            key = cv.waitKey(self.rate) & 0xFF
            if key in self._actions["quit"]:
                return

        if loop:
            return self.play(window_name, framefunc, loop)
