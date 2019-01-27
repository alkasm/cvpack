import cv2


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
