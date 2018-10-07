import cv2
import numpy as np
from typing import Callable


_make_actions = lambda chars: set(ord(c) for c in chars)


class VideoPlayer:

    _actions = {'quit': _make_actions('\x1bq')}
    
    def __init__(self, cap):
        self.cap = cap
        self.rate = int(1000.0/self.cap.fps)

    def play(self, window_name='VideoPlayer', framefunc=None, loop=False):
        """Plays through the video file with OpenCV's imshow()."""

        framefunc = framefunc or (lambda frame: frame)

        cont = True
        while cont:
            cont = loop
            for frame in self.cap:
                cv2.imshow(window_name, framefunc(frame))
                key = cv2.waitKey(self.rate) & 0xFF
                if key in self._actions['quit']:
                    cont = False
                    break


class VideoPlayerWithController:
    """Mimics YouTube's keyboard controls for playing a video via OpenCV's imshow.

    Actions
    -------

    play/pause:
        [k] or [space]: toggle
    close player:
        [q] or [esc]
    seek:
        [j]: -10 seconds
        [l]: +10 seconds
    scrub: skip through the video
        [0-9]: skip to 0%, 10%, 20%, ..., 90% through the video
    frames: go to (only when paused)
        [<]: previous frame
        [>]: next frame
    fullscreen:
        [f]: toggle fullscreen/original video size
    """
    
    _actions = {
        'pause': _make_actions('k '),
        'quit': _make_actions('\x1bq'),
        'seek': _make_actions('jl'),
        'scrub': _make_actions('0123456789'),
        'frames': _make_actions(',.')
    }

    def __init__(self, cap):
        self.cap = cap
        self.rate = int(1000.0/self.cap.fps)

    def _scrub(self, key):
        self.cap.pos_frames = int(chr(key)) * self.cap.frame_count / 10

    def _execute_action(self, key):
        try:
            if key == ord(','): self.cap.pos_frames -= 2
            elif key == ord('.'): pass
            elif key == ord('j'): self.cap.pos_msec -= 10*1000
            elif key == ord('l'): self.cap.pos_msec += 10*1000
            elif key in self._actions['scrub']: self._scrub(key)
        except AttributeError as e:
            print(e)
        return ord(' ')

    def _pause_actions(self, key):
        key = cv2.waitKey() & 0xFF
        while key not in set().union(*self._actions.values()):
            key = cv2.waitKey() & 0xFF
        if key in self._actions['pause']:
            return 0xFF
        elif key in self._actions['quit']:
            return ord('q')
        return self._execute_action(key)

    def _play_actions(self, key):
        if key in (self._actions['scrub'] | self._actions['seek']):
            self._execute_action(key)

    def _setup_window(self, window_name):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, int(self.cap.frame_width), int(self.cap.frame_height))

    def play(self, window_name: str='VideoPlayer', framefunc: Callable[[np.ndarray], np.ndarray]=None,
             autoplay: bool=False, loop: bool=False) -> None:
        """Plays through the video file with OpenCV's imshow()."""

        self._setup_window(window_name)
        framefunc = framefunc or (lambda frame: frame)
        key = 0xFF if autoplay else ord(' ')

        anyaction = set().union(*self._actions.values())

        cont = True
        while cont:
            cont = loop
            for frame in self.cap:
                cv2.imshow(self.window_name, framefunc(frame))
                if key in self._actions['pause']:          # pause
                    key = self._pause_actions(key)
                    continue
                elif key in self._actions['quit']:
                    cont = False
                    break
                self._play_actions(key)
                key = cv2.waitKey(self.rate) & 0xFF

