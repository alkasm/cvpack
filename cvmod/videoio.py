from typing import Any, Callable, Iterable, Optional, Type, Union, cast
from pathlib import Path
import cv2 as cv
import numpy as np
from .cvtypes import Size


class VideoCaptureProperty:
    """Descriptors to alias cap.get(cv.CAP_PROP_*) and cap.set(cv.CAP_PROP_*, value).

    Raises AttributeError when setting if that property is not supported.
    """

    _set_err: str = (
        "Unable to set the property {p}. The property might not be supported by\n"
        "the backend used by the cv.VideoCapture() instance."
    )

    def __init__(self, prop: str):
        self.prop = prop

    def __get__(
        self, obj: "VideoCapture", objtype: Optional[Type["VideoCapture"]] = None
    ) -> float:
        return cast(float, obj.cap.get(self.prop))

    def __set__(self, obj: "VideoCapture", value: float) -> None:
        if not obj.cap.set(self.prop, value):
            raise AttributeError(self._set_err.format(p=self.prop))


class VideoCapture:
    """An adapter for `cv.VideoCapture`, giving a more Pythonic interface."""

    pos_msec = VideoCaptureProperty(cv.CAP_PROP_POS_MSEC)
    pos_frames = VideoCaptureProperty(cv.CAP_PROP_POS_FRAMES)
    pos_avi_ratio = VideoCaptureProperty(cv.CAP_PROP_POS_AVI_RATIO)
    frame_width = VideoCaptureProperty(cv.CAP_PROP_FRAME_WIDTH)
    frame_height = VideoCaptureProperty(cv.CAP_PROP_FRAME_HEIGHT)
    fps = VideoCaptureProperty(cv.CAP_PROP_FPS)
    fourcc = VideoCaptureProperty(cv.CAP_PROP_FOURCC)
    frame_count = VideoCaptureProperty(cv.CAP_PROP_FRAME_COUNT)
    format = VideoCaptureProperty(cv.CAP_PROP_FORMAT)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.cap = cv.VideoCapture(*args, **kwargs)
        if not self.cap.isOpened():
            raise ValueError(
                f"Unable to open video source: args: {args} kwargs: {kwargs}"
            )

    def __getattr__(self, key: str) -> Any:
        return getattr(self.cap, key)

    def __iter__(self) -> Iterable[np.ndarray]:
        """Iterate through frames in the video."""
        noread = (False, None)
        if self.cap.isOpened():
            for _, frame in iter(self.cap.read, noread):
                yield frame

    def __enter__(self) -> "VideoCapture":
        """Enter the context manager."""
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        """Releases the video capture object on exiting the context manager."""
        self.cap.release()


class VideoWriter:
    filename: str
    fourcc: int
    fps: float

    _nowriter = object()

    def __init__(
        self,
        filename: Union[Path, str],
        fourcc: Union[int, Iterable[str]] = "mp4v",
        fps: float = 30,
        frameSize: Any = None,
        **kwargs: Any,
    ) -> None:
        self.filename = str(filename)
        self.fourcc = (
            fourcc if isinstance(fourcc, int) else cv.VideoWriter_fourcc(*fourcc)
        )
        self.fps = fps
        self._kwargs = kwargs

        # wait to create writer based on first frame if size is not provided
        self._writer = (
            self._nowriter if frameSize is None else self._makewriter(frameSize)
        )

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.release()

    def _makewriter(self, frame_size: Any) -> cv.VideoWriter:
        return cv.VideoWriter(
            filename=self.filename,
            fourcc=self.fourcc,
            fps=self.fps,
            frameSize=frame_size,
            **self._kwargs,
        )

    def write(self, frame: np.ndarray) -> bool:
        try:
            return cast(bool, self._writer.write(frame))
        except AttributeError as e:
            if self._writer is self._nowriter:
                size = Size.from_image(frame)
                self._writer = self._makewriter(size)
                return self.write(frame)
            else:
                raise e

    def release(self) -> None:
        try:
            self._writer.release()
        except AttributeError as e:
            if self._writer is not self._nowriter:
                raise e


class VideoPlayer:
    cap: VideoCapture
    rate: float

    _actions = {"quit": {ord("\x1b"), ord("q")}}

    def __init__(self, cap: VideoCapture) -> None:
        self.cap = cap
        self.rate = int(1000 / self.cap.fps)

    def play(
        self,
        window_name: str = "VideoPlayer",
        framefunc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        loop: bool = False,
    ) -> None:
        """Plays through the video file with OpenCV's imshow()."""
        frames = (
            self.cap.__iter__()
            if framefunc is None
            else map(framefunc, self.cap.__iter__())
        )
        for frame in frames:
            cv.imshow(window_name, frame)
            key = cv.waitKey(self.rate) & 0xFF
            if key in self._actions["quit"]:
                return

        if loop:
            return self.play(window_name, framefunc, loop)
