import base64
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import cast, Any, Optional, Tuple, Union
import urllib.request
import webbrowser
import cv2 as cv
import numpy as np

__all__ = [
    "imread_url",
    "imwrite",
    "imshow",
    "imshow_jupyter",
    "imshow_browser",
    "color_labels",
    "normalize",
    "enlarge",
    "add_grid",
]


def imread(imgpath: Union[Path, str], *args: Any, **kwargs: Any) -> np.ndarray:
    """Reads an image, providing helpful errors on failed reads.

    Allows pathlib.Path objects as well as strings for the imgpath.
    """
    img = cv.imread(str(imgpath), *args, **kwargs)
    if img is None:
        p = Path(imgpath)
        if p.exists():
            raise ValueError("Image is empty!")
        raise FileNotFoundError(f"Image path {p.absolute()} doesn't exist!")

    return img


def imread_url(url: str, *args: Any, **kwargs: Any) -> np.ndarray:
    """Reads an image from a given url.

    Additional args and kwargs passed onto cv.imdecode()
    """
    r = urllib.request.urlopen(url)
    content_type = r.headers.get_content_maintype()
    if content_type != "image":
        raise ValueError(f"Unknown content type {content_type}.")
    buf = np.frombuffer(r.read(), np.uint8)
    return cv.imdecode(buf, *args, **kwargs)


def imwrite(
    imgpath: Union[Path, str], img: np.ndarray, *args: Any, **kwargs: Any
) -> bool:
    """Writes an image, providing helpful errors on failed writes.

    Allows pathlib.Path objects as well as strings for the imgpath.
    Will create the directories included in the imgpath if they don't exist.

    Additional args and kwargs passed to cv.imwrite().
    """
    if img is None:
        raise ValueError("Image is empty!")
    Path(imgpath).parent.mkdir(parents=True, exist_ok=True)
    imgpath = str(imgpath)
    return cast(bool, cv.imwrite(imgpath, img, *args, **kwargs))


def imshow(img: np.ndarray, wait: int = 0, window_name: str = "") -> int:
    """Combines cv.imshow() and cv.waitkey(), and checks for bad image reads."""
    if img is None:
        raise ValueError(
            "Image is empty; ensure you are reading from the correct path."
        )
    cv.imshow(window_name, img)
    return cast(int, cv.waitKey(wait) & 0xFF)


def imshow_jupyter(img: np.ndarray) -> None:
    """Shows an image in a Jupyter notebook.

    Raises ValueError if img is None or if img cannot be encoded.
    """
    if img is None:
        raise ValueError("Image has no data (img is None).")

    success, encoded = cv.imencode(".png", img)
    if not success:
        raise ValueError("Error encoding image.")

    try:
        from IPython.display import Image, display

        display(Image(encoded))
    except ImportError:
        print("You must have IPython installed to use the IPython display.")
        raise


def imshow_browser(
    img: np.ndarray, host: str = "localhost", port: int = 32830, route: str = "imshow"
) -> None:
    """Display an image in a browser.

    Spins up a single-request server to serve the image.
    Opens the browser to make that request, then shuts down the server.
    """

    class ImshowRequestHandler(_ImshowRequestHandler):
        imshow_img = img
        imshow_route = route

    server = HTTPServer((host, port), ImshowRequestHandler)
    with ThreadPoolExecutor(max_workers=1) as executor:
        # handle_request() blocks, so submit in an executor.
        # the browser can open the window and get served the image,
        # at which point the submitted task is completed.
        executor.submit(server.handle_request)
        webbrowser.open_new(f"http://{host}:{port}/{route}")


def _html_imshow(img: np.ndarray) -> str:
    success, encoded_img = cv.imencode(".png", img)

    html = """<html>
<title>cvpack/imshow</title>
<head> 
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body style="background-color:#16151a;margin:0;">
<div style="display:flex;justify-content:center;align-items:center;height:100vh;">
<img src="data:image/png;base64,{encoded_img}"/>
</div>
</body>
</html>
"""

    b64_encoded_img = base64.b64encode(encoded_img).decode() if success else ""
    return html.format(encoded_img=b64_encoded_img)


class _ImshowRequestHandler(BaseHTTPRequestHandler):
    imshow_route: str = ""
    imshow_img: Optional[np.ndarray] = None

    # handle GET request from browser
    def do_GET(self) -> None:
        self.send_response(200)
        if self.path == f"/{self.imshow_route}":
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(_html_imshow(self.imshow_img).encode())
        return

    # remove logging statements by returning nothing here
    def log_message(self, format: str, *args: Any) -> None:
        return


def color_labels(labels: np.ndarray) -> np.ndarray:
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0
    return labeled_img


def normalize(img: np.ndarray) -> np.ndarray:
    return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)


def enlarge(img: np.ndarray, scale: int = 10) -> np.ndarray:
    return cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)


def add_grid(
    img: np.ndarray,
    spacing: int = 10,
    color: Union[float, Tuple[float, float, float]] = 200,
) -> np.ndarray:
    viz = img.copy()
    h, w = img.shape[:2]

    partial_grid = np.zeros((spacing, spacing), dtype=bool)
    partial_grid[0, :] = True
    partial_grid[:, 0] = True
    gridlines = np.tile(partial_grid, (h // spacing, w // spacing))
    viz[gridlines] = color

    pad_sizes = ((0, 1), (0, 1), (0, 0)) if len(img.shape) == 3 else ((0, 1), (0, 1))
    viz = np.pad(viz, pad_sizes)
    viz[-1, :] = color
    viz[:, -1] = color

    return viz
