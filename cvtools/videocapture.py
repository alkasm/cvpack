import cv2


"""Constants"""


VIDEO_CAPTURE_PROPS_LIST = [
    'CAP_PROP_POS_MSEC',
    'CAP_PROP_POS_FRAMES',
    'CAP_PROP_POS_AVI_RATIO',
    'CAP_PROP_FRAME_WIDTH',
    'CAP_PROP_FRAME_HEIGHT',
    'CAP_PROP_FPS',
    'CAP_PROP_FOURCC',
    'CAP_PROP_FRAME_COUNT',
    'CAP_PROP_FORMAT',
    'CAP_PROP_MODE',
    'CAP_PROP_BRIGHTNESS',
    'CAP_PROP_CONTRAST',
    'CAP_PROP_SATURATION',
    'CAP_PROP_HUE',
    'CAP_PROP_GAIN',
    'CAP_PROP_EXPOSURE',
    'CAP_PROP_CONVERT_RGB',
    'CAP_PROP_WHITE_BALANCE_BLUE_U',
    'CAP_PROP_RECTIFICATION',
    'CAP_PROP_MONOCHROME',
    'CAP_PROP_SHARPNESS',
    'CAP_PROP_AUTO_EXPOSURE',
    'CAP_PROP_GAMMA',
    'CAP_PROP_TEMPERATURE',
    'CAP_PROP_TRIGGER',
    'CAP_PROP_TRIGGER_DELAY',
    'CAP_PROP_WHITE_BALANCE_RED_V',
    'CAP_PROP_ZOOM',
    'CAP_PROP_FOCUS',
    'CAP_PROP_GUID',
    'CAP_PROP_ISO_SPEED',
    'CAP_PROP_BACKLIGHT',
    'CAP_PROP_PAN',
    'CAP_PROP_TILT',
    'CAP_PROP_ROLL',
    'CAP_PROP_IRIS',
    'CAP_PROP_SETTINGS',
    'CAP_PROP_BUFFERSIZE',
    'CAP_PROP_AUTOFOCUS',
]

VIDEO_CAPTURE_PROPS = {prop.split('CAP_PROP_')[-1].lower(): prop for prop in VIDEO_CAPTURE_PROPS_LIST}


"""Classes"""


class VideoCaptureProperty:
    """Descriptor for the video capture properties getter/setter."""

    _set_err = 'The property {p} is not supported by the backend used by the VideoCapture instance.'

    def __init__(self, name):
        self.name = name
        self.prop = getattr(cv2, VIDEO_CAPTURE_PROPS[name])
        self.__doc__ = "alias for cap.get(prop)/cap.set(prop, val), where prop = cv2." + VIDEO_CAPTURE_PROPS[name]

    def __get__(self, obj, objtype=None):
        return obj.get(self.prop)

    def __set__(self, obj, value):
        was_set = obj.set(self.prop, value)
        if not was_set:
            prop_name = 'CAP_PROP_' + self.name.upper()
            raise AttributeError(self._set_err.format(p=prop_name))


class VideoCapture(cv2.VideoCapture):

    def __new__(cls, *args, **kwargs):
        for name, prop in VIDEO_CAPTURE_PROPS.items():
            setattr(cls, name, VideoCaptureProperty(name))
        self = super().__new__(cls, *args, **kwargs)
        return self

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.isOpened():
            print('Warning: unable to open video source:', *args, **kwargs)

    def __del__(self):
        """Releases the video capture object."""
        self.release()

    def __iter__(self):
        """Resets the frame position to 0 at the start for repeatable iteration."""
        try:
            self.pos_frames = 0
        except AttributeError:
            pass
        while self.isOpened():
            success, frame = self.read()
            if success: yield frame
            else: break

    def __reversed__(self):
        """Iterates through the frames in reversed order, starting at the end."""
        self.pos_frames = self.frame_count - 1
        while self.pos_frames > 0:
            success, frame = self.read()
            if success: yield frame
            else: break
            self.pos_frames -= 2

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exctype, exc, exctrace):
        """Releases the video capture object on exiting the context manager."""
        self.release()

