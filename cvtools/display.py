import cv2


def imshow(img, wait=0, window_name=""):
    if img is None:
        raise ValueError(
            "Image is empty; ensure you are reading from the correct path."
        )
    cv2.imshow(window_name, img)
    return cv2.waitKey(wait) & 0xFF


def imshow_ipython(img):
    """Shows an image in a Jupyter notebook.
    Raises ValueError if img is None or if img cannot be encoded.
    """
    if img is None:
        raise ValueError("Image has no data (img is None).")

    success, encoded = cv2.imencode(".png", img)
    if not success:
        raise ValueError("Error encoding image.")

    from IPython.display import Image, display

    display(Image(encoded))


def imshow_components(labels, *args, **kwargs):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0
    return imshow(labeled_img, *args, **kwargs)


def imshow_autoscale(img, *args, **kwargs):
    scaled = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return imshow(scaled, *args, **kwargs)


def imshow_enlarged(img, scale=10, grid=True, color=200, wait=0, window_name=""):
    h, w = img.shape[:2]
    r = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    if grid:
        r = add_grid(img, scale, color)
    return imshow(r, wait, window_name)


def add_grid(img, scale=10, color=200):

    h, w = img.shape[:2]
    r = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    partial_grid = np.zeros((scale, scale), dtype=bool)
    partial_grid[0, :] = True
    partial_grid[:, 0] = True
    gridlines = np.tile(partial_grid, (h, w))
    r[gridlines] = color

    pad_sizes = ((0, 1), (0, 1), (0, 0)) if len(img.shape) == 3 else ((0, 1), (0, 1))
    r = np.pad(r, pad_sizes, "constant", constant_values)
    r[-1, :] = color
    r[:, -1] = color

    return r


# Display image in browser (for working in interpreters without a GUI)


from http.server import BaseHTTPRequestHandler, HTTPServer
import webbrowser
from concurrent.futures import ThreadPoolExecutor
import base64


def _html_imshow(img):
    success, encoded_img = cv2.imencode(".png", img)

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


def imshow_browser(img, server="localhost", port=32830, route="imshow"):
    """Display an image in a browser. 
    
    Spins up a server to serve the image for a single request.
    Opens the browser to make that request, then shuts down the server.
    """

    class ImshowRequestHandler(_ImshowRequestHandler):
        imshow_img = img
        imshow_route = route

    host = "localhost"
    port = 32830
    server = HTTPServer((host, port), ImshowRequestHandler)
    with ThreadPoolExecutor(max_workers=1) as executor:
        # handle_request() blocks, so submit in an executor.
        # the browser can open the window and get served the image,
        # at which point the submitted task is completed.
        future = executor.submit(server.handle_request)
        webbrowser.open_new(f"http://{host}:{port}/{route}")
