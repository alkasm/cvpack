from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import webbrowser
import base64
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import cv2 as cv
import numpy as np


def imread(imgpath, *args, **kwargs):
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


def imread_web(url, *args, **kwargs):
    """Reads an image from the web."""
    r = urllib.request.urlopen(url)
    if r.headers.get_content_maintype() != "image":
        raise ValueError(f"Unknown content type {content_type}.")
    buf = np.frombuffer(r.read(), np.uint8)
    return cv.imdecode(buf, *args, **kwargs)


def imwrite(imgpath, img, *args, **kwargs):
    """Writes an image, providing helpful errors on failed writes.

    Allows pathlib.Path objects as well as strings for the imgpath.
    Will create the directories included in the imgpath if they don't exist.
    """
    if img is None:
        raise ValueError("Image is empty!")
    Path(imgpath).parent.mkdir(parents=True, exist_ok=True)
    imgpath = str(imgpath)
    return cv.imwrite(imgpath, img, *args, **kwargs)


def imshow(img, wait=0, window_name=""):
    if img is None:
        raise ValueError(
            "Image is empty; ensure you are reading from the correct path."
        )
    cv.imshow(window_name, img)
    return cv.waitKey(wait) & 0xFF


def imshow_ipython(img):
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


def imshow_components(labels, *args, **kwargs):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0
    return imshow(labeled_img, *args, **kwargs)


def imshow_autoscale(img, *args, **kwargs):
    scaled = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    return imshow(scaled, *args, **kwargs)


def imshow_enlarged(img, scale=10, grid=True, color=200, wait=0, window_name=""):
    if grid:
        r = _add_grid(img, scale, color)
    else:
        r = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
    return imshow(r, wait, window_name)


def _add_grid(img, scale=10, color=200):

    h, w = img.shape[:2]
    r = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)

    partial_grid = np.zeros((scale, scale), dtype=bool)
    partial_grid[0, :] = True
    partial_grid[:, 0] = True
    gridlines = np.tile(partial_grid, (h, w))
    r[gridlines] = color

    pad_sizes = ((0, 1), (0, 1), (0, 0)) if len(img.shape) == 3 else ((0, 1), (0, 1))
    r = np.pad(r, pad_sizes, "constant", color)
    r[-1, :] = color
    r[:, -1] = color

    return r


def _html_imshow(img):
    success, encoded_img = cv.imencode(".png", img)

    html = """
<html>
<title>cvtools/imshow</title>
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
    imshow_route = ""
    imshow_img = None

    # handle GET request from browser
    def do_GET(self):
        self.send_response(200)
        if self.path == f"/{self.imshow_route}":
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(_html_imshow(self.imshow_img).encode())
        return

    # remove logging statements by returning nothing here
    def log_message(self, format, *args):
        return


def imshow_browser(img, host="localhost", port=32830, route="imshow"):
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
