import cv2


VIDEO_CAPTURE_PROPS_LIST = [
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

VIDEO_CAPTURE_PROPS = {
    prop.split("CAP_PROP_").pop().lower(): prop for prop in VIDEO_CAPTURE_PROPS_LIST
}


def add_props(props):
    def deco(cls):
        for prop in props:
            setattr(cls, prop, VideoCaptureProperty(prop))
        return cls

    return deco


class VideoCaptureProperty:

    _set_err = (
        "The property {p} is not supported by\n"
        "the backend used by the cv2.VideoCapture() instance."
    )
    _docstring = (
        "Alias for cap.get(cv2.{p}) and cap.set(cv2.{p}, value).\n"
        "Raises AttributeError when setting if that property is not supported.\n"
    )

    def __init__(self, name):
        self.name = name
        self.prop = getattr(cv2, VIDEO_CAPTURE_PROPS[name])
        self.__doc__ = self._docstring.format(p=VIDEO_CAPTURE_PROPS[name])

    def __get__(self, obj, objtype=None):
        return obj.get(self.prop)

    def __set__(self, obj, value):
        if not obj.set(self.prop, value):
            raise AttributeError(self._set_err.format(p=VIDEO_CAPTURE_PROPS[self.name]))


@add_props(VIDEO_CAPTURE_PROPS.keys())
class VideoCapture(cv2.VideoCapture):
    """An adapter for `cv2.VideoCapture`, giving a more Pythonic interface."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.isOpened():
            raise ValueError("Unable to open video source:", *args, **kwargs)

    def __iter__(self):
        """Resets the frame position to 0 at the start for repeatable iteration."""
        try:
            self.pos_frames = 0
        except AttributeError:
            pass
        while self.isOpened():
            success, frame = self.read()
            if success:
                yield frame
            else:
                return

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exctype, exc, exctrace):
        """Releases the video capture object on exiting the context manager."""
        self.release()


class VideoPlayer:

    _actions = {"quit": set(map(ord, "\x1bq"))}

    def __init__(self, cap):
        self.cap = cap
        self.rate = int(1000 / self.cap.fps)

    def play(self, window_name="VideoPlayer", framefunc=None, loop=False):
        """Plays through the video file with OpenCV's imshow()."""

        framefunc = framefunc or (lambda frame: frame)

        for frame in self.cap:
            cv2.imshow(window_name, framefunc(frame))
            key = cv2.waitKey(self.rate) & 0xFF
            if key in self._actions["quit"]:
                return

        if loop:
            return self.play(window_name, framefunc, loop)
