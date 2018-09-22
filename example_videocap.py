import cv2
from cvtools import VideoCapture, BasicVideoPlayer

with VideoCapture(0) as cap:
    player = BasicVideoPlayer(cap)
    player.play(loop=True, framefunc=lambda f: cv2.flip(f, 1))