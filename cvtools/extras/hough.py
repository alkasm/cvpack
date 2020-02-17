import cv2
import numpy as np
from collections import defaultdict
from bisect import bisect


# Constants


RHOTHETA = "rhotheta"
ENDPOINT = "endpoint"
ERRORVAL = np.nan


# Utility Functions


def linetype(line):
    """Returns 'rhotheta' if line is defined by two variables,
    i.e. [[rho, theta]], or 'endpoint' if line is defined by four
    variables, i.e. [[x1, y1, x2, y2]].

    Parameters
    ----------
    line : np.ndarray
        Either np.array([[rho, theta]]) or np.array([[x1, y1, x2, y2]])
        where theta is in radians.

    Returns
    -------
    linetype : str
        'rhotheta' if line contains two vars; 'endpoint' if four.

    Example
    -------
    >>> import numpy as np
    >>> rho, theta = 5, np.pi/4
    >>> line_rhotheta = np.array([[rho, theta]])
    >>> x1, y1, x2, y2 = 0, 0, 50, 100
    >>> line_endpoint = np.array([[x1, y1, x2, y2]])
    >>> linetype(line_rhotheta)
    'rhotheta'
    >>> linetype(line_endpoint)
    'endpoint'
    """

    if len(line[0]) == 2:
        return RHOTHETA
    elif len(line[0]) == 4:
        return ENDPOINT
    else:
        raise TypeError


def lineangle(line):
    """Returns the angle (in radians) of the line in [0, 2π).

    Lines in rho-theta form are only defined with theta in [0, π)
    but allows negative rho values; this function takes into account
    the sign of rho to determine the angle in [0, 2π).

    Similarly, lines in endpoint form will return different angles if
    their endpoints are flipped.

    Parameters
    ----------
    line : np.ndarray
        Either np.array([[rho-theta]]) or np.array([[x1, y1, x2, y2]])
        where `theta` is in radians.

    Returns
    -------
    angle : float
        Angle in radians in the interval [0, 2π) of the line.

    Example
    -------
    >>> import numpy as np
    >>> rho, theta = -5, 3*np.pi/4
    >>> line_rhotheta = create_line(rho, theta)
    >>> x1, y1, x2, y2 = 0, 0, 50, 50
    >>> line_endpoint = create_line(x1, y1, x2, y2)
    >>> print('lineangle(line_rhotheta):', lineangle(line_rhotheta))
    lineangle(line_rhotheta): 0.785398163397
    >>> print('lineangle(line_endpoint):', lineangle(line_endpoint))
    lineangle(line_endpoint): 0.785398163397
    >>> print('radian(45 degrees):', np.pi/4)
    radian(45 degrees): 0.7853981633974483
    """

    _linetype = linetype(line)
    if _linetype == RHOTHETA:
        # actual angle of line is perpendicular to theta
        rho, theta = line[0]
        if rho >= 0:
            angle = theta + np.pi / 2
        else:
            angle = theta - np.pi / 2
        return angle % (2 * np.pi)
    elif _linetype == ENDPOINT:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1)
        return angle % (2 * np.pi)
    else:
        raise TypeError


def isparallel(line1, line2, tol=None):
    """Boolean to check if two lines are parallel.

    Parameters
    ----------
    line1 : np.ndarray
        Either rho-theta or endpoint form.
    line2 : np.ndarray
        Either rho-theta or endpoint form; does not need to be the same
        as line1.
    tol : float
        An optional tolerance value for parallelism check. If tol=0,
        the function checks for exact parallelism (that is, the lines
        have the exact same angle to machine precision). By default,
        the tolerance is a single arc-second: tol=np.pi/(180*60*60).

    Returns
    -------
    isparallel : bool
        True if the difference in angle between each line is less than
        tol, False otherwise. 

    Example
    -------
    >>> line1 = create_line(25, np.pi/4)
    >>> line2 = create_line(3, np.pi/4)
    >>> isparallel(line1, line2)
    True
    """
    tol = np.pi / (180 * 60 * 60) if tol == None else tol

    angle1 = lineangle(line1) % np.pi
    angle2 = lineangle(line2) % np.pi
    diff = abs(angle1 - angle2)
    diff = min(diff, np.pi - diff)
    return diff < tol


def length(line):
    """Get the length of a 2-D line segment using Euclidean distance.

    Parameters
    ----------
    line : np.ndarray
        A line in endpoint form.
    
    Returns
    -------
    length : float
        Euclidean length of the line.

    Example
    -------
    >>> line = create_line(0, 0, 100, 100)
    >>> length(line)
    141.42135623730951
    """
    x1, y1, x2, y2 = line[0]
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def create_line_rhotheta(rho, theta):
    """Create a line in rho-theta form.

    Parameters
    ----------
    rho : int
        The radius of the circle centered at (0, 0) which the line is
        tangent to.
    theta : float
        The angle (in radians) about the origin; the line is defined
        perpendicularly to this angle.

    Returns
    -------
    line : np.ndarray
        A line in the form np.array([[rho, theta]]). The angle returned
        will be in [0, π), with negative rho values for angles in the
        interval [π, 2π).

    Example
    -------
    >>> import numpy as np
    >>> rho, theta = 5, np.pi/4
    >>> create_line_rhotheta(rho, theta)
    array([[ 5.        ,  0.78539816]])
    >>> create_line(3, 5*np.pi/4)
    array([[-3.        ,  0.78539816]])
    """
    theta = theta % (2 * np.pi)
    if theta < np.pi:
        return np.array([[rho, theta]])
    return np.array([[-rho, theta % np.pi]])


def create_line_endpoint(x1, y1, x2, y2):
    """Create a line in endpoint form.

    Parameters
    ----------
    x1, y1, x2, y2 : numeric
        The endpoints of the line are (x1, y1) and (x2, y2). Points
        can be specified with ints or float subpixel points.

    Returns
    -------
    line : np.ndarray
        A line in the form np.array([[x1, y1, x2, y2]]).

    Example
    -------
    >>> x1, y1, x2, y2 = 0, 0, 500, 500
    >>> create_line_endpoint(x1, y1, x2, y2)
    array([[  0,   0, 500, 500]])
    """
    return np.array([[x1, y1, x2, y2]])


def create_line(*args):
    """Create a line in rho-theta or endpoint form, depending on the
    number of arguments specified.

    Parameters
    ----------
    *args : numeric
        Either two numbers specifying `rho, theta` or four numbers
        specifying `x1, y1, x2, y2`. 

    Returns
    -------
    line : np.ndarray
        A line in rho-theta form `np.array([[rho, theta]])` or endpoint
        form `np.array([[x1, y1, x2, y2]])`. If it is in rho-theta form,
        the angle returned will be in `[0, π)`, with negative `rho` values
        for angles input in `[π, 2π)`.

    Example
    -------
    >>> import numpy as np
    >>> rho, theta = 5, np.pi/4
    >>> create_line(rho, theta)
    array([[ 5.        ,  0.78539816]])
    >>> x1, y1, x2, y2 = 0, 0, 500, 500
    >>> create_line(x1, y1, x2, y2)
    array([[  0,   0, 500, 500]])
    
    """
    if len(args) == 2:
        return create_line_rhotheta(*args)
    elif len(args) == 4:
        return create_line_endpoint(*args)
    else:
        raise TypeError


# Distance Functions


def point_line_dist(point, line):
    """See
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
    """

    _linetype = linetype(line)
    if linetype(line) == RHOTHETA:
        line = convert(line)
    for x1, y1, x2, y2 in line:
        for x0, y0 in point:
            numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            denominator = length(line)
            if denominator < 1:
                print("Error: line length less than a single pixel.")
                return ERRORVAL
    return numerator / denominator


def point_point_dist(point1, point2):
    return length(np.array([[*point1, *point2]]))


# Conversion Functions


def endpoint(line, bbox=[0, 0, 1e5, 1e5]):
    """Converts a line from rho-theta to endpoint form.

    Endpoints are given by the intersection to the input bounding box.

    Parameters
    ----------
    line : np.ndarray
        A line in rho-theta form.
    bbox : list, optional
        A bounding box defined by the top-left and bottom-right points
        or in an alternative view by the left-top-right-bottom walls,
        in the form [x1, y1, x2, y2]; default [0, 0, 1e5, 1e5].
        Intersections are found between the input line and the border
        lines of the bounding box. Only the two points touching the 
        border of the bounding box are returned. For a specific image
        height and width, set bbox=[0, 0, w, h].
    
    Returns
    -------
    line : np.ndarray
        Returns an endpoint line with endpoints at the intersection
        with bounding box borders.
    """
    x1, y1, x2, y2 = bbox
    border_lines = [[[x1, np.pi / 2]], [[x2, np.pi / 2]], [[y1, 0]], [[y2, 0]]]

    interx = [intersection(line, border_lines[i]) for i in range(4)]

    bounded_interx = []
    for point in interx:
        for x, y in point:
            if (x1 <= x <= x2) and (y1 <= y <= y2):
                bounded_interx.extend([x, y])
        if len(bounded_interx) == 4:
            # in case a line hits the corner exactly
            break
    if len(bounded_interx) != 4:
        # line is outside the bounding box
        return [[[ERRORVAL, ERRORVAL]]] * 4
    return np.array([bounded_interx])


def rhotheta(line):
    """Converts a line from endpoint to rho-theta form.

    Infinitely extending rho-theta lines are returned.

    Parameters
    ----------
    line : np.ndarray
        A line in endpoint form.
    
    Returns
    -------
    line : np.ndarray
        An infinitely extending rho-theta line.
    """
    theta = lineangle(line)
    sign = 1 if theta < np.pi else -1
    theta = theta % np.pi
    rho = sign * point_line_dist([[0, 0]], line)
    return np.array([[rho, theta]])


def convert(line, bbox=[0, 0, 1e5, 1e5]):
    """Converts a line between rho-theta and endpoint form.

    When rho-theta lines are input, endpoints are given by the
    intersection to the input bounding box. When endpoint lines
    are input, infinitely extending rho-theta lines are returned.

    Parameters
    ----------
    line : np.ndarray or list
        A line in rho-theta or endpoint form, to be converted
        to the other; *or* a list (possibly nested) of such lines.
    bbox : list, optional
        Used only when converting from rho-theta to endpoint lines. 
        A bounding box defined by the top-left and bottom-right points
        or in an alternative view by the left-top-right-bottom walls,
        in the form [x1, y1, x2, y2]; default [0, 0, 1e5, 1e5].
        Intersections are found between the input line and the border
        lines of the bounding box. Only the two points touching the 
        border of the bounding box are returned. For a specific image
        height and width, set bbox=[0, 0, w, h].
    
    Returns
    -------
    line : np.ndarray
        If the input is a rho-theta line, returns an endpoint line
        with intersections at the bounding box borders. If the input
        line is an endpoint line, returns a rho-theta line. If the 
        input is a list of lines, then returns the converted list.
    """
    if isinstance(line, list):
        return [convert(l, bbox) for l in line]
    _linetype = linetype(line)
    if _linetype == RHOTHETA:
        return endpoint(line, bbox)
    elif _linetype == ENDPOINT:
        return rhotheta(line)
    else:
        raise TypeError


def pointslope(line, x=0, tol=None):
    """Convert from rho-theta or endpoint line to point-slope form.

    Parameters
    ----------
    line : np.ndarray
        A rho-theta or endpoint line.
    x : numeric, optional
        The point on the line returned will pass through the vertical
        line at x. By default, x=0, rendering this function identical
        except in output format to slopeintercept(line).
    tol : float, optional
        Tolerance for checking verticalism. See docs of isparallel().

    Returns
    -------
    line : np.ndarray
        A line in the form np.array([[x, y, m]]) where (x, y) is a
        point on the line and m is the slope of the line. In the case
        of a vertical line, [[x', np.nan, np.nan]] will be returned,
        where x' is the location of the line. In any other case, the
        returned line will pass through the vertical line defined by x.
    """
    _linetype = linetype(line)

    if _linetype == RHOTHETA:
        rho, theta = line[0]
        vline = create_line(x, 0)
        if isparallel(line, vline):
            return np.array([[rho, ERRORVAL, ERRORVAL]])
        m = np.sign(rho) * np.tan(theta + np.pi / 2)
    elif _linetype == ENDPOINT:
        x1, y1, x2, y2 = line[0]
        if isparallel(line, vline):
            return np.array([[x1, ERRORVAL, ERRORVAL]])
        m = (y2 - y1) / (x2 - x1)
        vline = create_line(x, 0, x, 1)

    interx = intersection(line, vline)
    x, y = interx[0]

    return np.array([[x, y, m]])


def slopeintercept(line, tol=None):
    """Convert from rho-theta or endpoint line to slope-intercept form.

    Parameters
    ----------
    line : np.ndarray
        A rho-theta or endpoint line.
    tol : float, optional
        Tolerance for checking verticalism. See docs of isparallel().

    Returns
    -------
    line : np.ndarray
        A line in the form np.array([[m, b]]) where m is the slope of
        the line and b is the y-intercept. In the case of a vertical
        line, [[np.nan, np.nan]] will be returned; in any other case,
        the returned line will follow the Cartesian equation y=mx+b.
    """
    x, y, m = pointslope(line, tol)
    if x == 0:
        return np.array([[m, y]])
    else:
        return np.array([[ERRORVAL, ERRORVAL]])


# Segmenting Functions


def segment_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Computes the angle of each line and uses k-means on the coordinates
    of the angle on the unit circle to segment k angles inside lines.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get("criteria", (default_criteria_type, 10, 1.0))
    flags = kwargs.get("flags", cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get("attempts", 10)

    # returns angles in [0, π)
    angles = [lineangle(line) % np.pi for line in lines]
    # stretch angles to the full range and map to coords
    pts = np.array(
        [[[np.cos(2 * angle), np.sin(2 * angle)]] for angle in angles], dtype=np.float32
    )
    # run kmeans on unit coordinates of the angle
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def segment_angle_range(lines, step=np.pi / 2):
    """Groups lines based on angle.

    Chops up the interval [0, π) into ceil(π/step) bins and assigns
    each line to a bin. The first bin is only half-width and merged
    with the last bin so lines that are nearly horizontal are counted
    in the same bin.
    """
    angle_cuts = np.arange(step / 2, np.pi, step=step)
    segmented = defaultdict(list)
    for line in lines:
        angle_bin = bisect(angle_cuts, lineangle(line) % np.pi) % len(angle_cuts)
        segmented[angle_bin].append(line)
    segmented = list(segmented.values())
    return segmented


def segment_angle_linspace(lines, num=2):
    """Groups lines based on angle.

    Chops up the interval [0, π) into num bins and assigns each line
    to a bin. The first bin is only half-width and merged with the last
    bin so lines that are nearly horizontal are counted in the same bin.
    """
    step = np.pi / num
    return segment_angle_range(lines, step)


# Intersection Functions


def rhotheta_intersection(line1, line2, tolerance=1e-6, subpixel=False):
    """Finds the intersection of two lines in rho, theta form.

    Returns closest integer pixel locations. Returns nan if lines are
    parallel or very close to parallel.

    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]

    parallelism = abs(theta1 - theta2)
    if parallelism < tolerance:
        return [[ERRORVAL, ERRORVAL]]

    # Ax = b    linear system
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)

    if subpixel:
        return [[x0, y0]]
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def endpoint_intersection(line1, line2, tolerance=1e-6, subpixel=False):
    """Finds the intersection of two lines in endpoint form.

    Returns closest integer pixel locations. Returns nan if lines are
    parallel or very close to parallel.

    See https://stackoverflow.com/a/383527/5087436
    """

    parallelism = abs(_line_angle(line1) - _line_angle(line2))
    if parallelism < tolerance:
        return [[ERRORVAL, ERRORVAL]]

    # extract points
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    # compute determinant
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    x0 = (
        (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denominator
    y0 = (
        (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denominator

    if subpixel:
        return [[x0, y0]]
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def intersection(line1, line2, tolerance=1e-6, subpixel=False):
    """Finds the intersection of two lines.

    Returns closest integer pixel locations. Returns np.nan if lines are
    parallel or very close to parallel, or if two lines are not defined
    in rho, theta or endpoint form.
    """
    linetype1 = linetype(line1)
    linetype2 = linetype(line2)

    if linetype1 == linetype2 == RHOTHETA:
        return rhotheta_intersection(line1, line2, tolerance, subpixel)
    elif linetype1 == linetype2 == ENDPOINT:
        return endpoint_intersection(line1, line2, tolerance, subpixel)
    else:
        print("Line type error; nan intersection returned")
        return [[ERRORVAL, ERRORVAL]]


# Drawing Functions


def draw_lines(img, lines, color=None, thickness=1):
    h, w = img.shape[:2]
    color = color or (0, 255, 0) if len(img.shape) == 3 else 255
    _linetype = linetype(lines[0])
    if _linetype == RHOTHETA:
        lines = [convert(line, bbox=[0, 0, w, h]) for line in lines]
    for line in lines:
        for x1, y1, x2, y2 in line:
            if ERRORVAL in [x1, y1, x2, y2]:
                print("Line {} has np.nan vals; not drawing.".format(line))
                continue
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def mark_endpoints(img, lines, color=None, markerType=cv2.MARKER_CROSS, markerSize=5):
    color = color or (0, 255, 255) if len(img.shape) == 3 else 255
    for line in lines:
        for x1, y1, x2, y2 in line:
            if ERRORVAL in [x1, y1, x2, y2]:
                print("Line {} has np.nan vals; not drawing.".format(line))
                continue
            cv2.drawMarker(img, (x1, y1), color, markerType, markerSize)
            cv2.drawMarker(img, (x2, y2), color, markerType, markerSize)
    return img


def mark_intersections(
    img, intersections, color=None, markerType=cv2.MARKER_TRIANGLE_UP, markerSize=5
):
    color = color or (0, 255, 255) if len(img.shape) == 3 else 255
    for point in intersections:
        for x, y in point:
            if ERRORVAL in [x, y]:
                print("Point {} has np.nan vals; not drawing.".format(point))
                continue
            cv2.drawMarker(img, (x, y), color, markerType, markerSize)
    return img
