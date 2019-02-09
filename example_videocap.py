import cv2
from cvtools import VideoCapture, VideoPlayer

with VideoCapture(0) as cap:
    player = VideoPlayer(cap)
    player.play(loop=True, framefunc=lambda f: cv2.flip(f, 1))
