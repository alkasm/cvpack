import cv2
import numpy as np


TWO_PI = 2 * np.pi


def kmeans_periodic(columns, intervals, data, *args, **kwargs):
    """Runs kmeans with periodicity in a subset of dimensions.
    
    Transforms columns with periodicity on the specified intervals into two
    columns with coordinates on the unit circle for kmeans. After running
    through kmeans, the centers are transformed back to the range specified
    by the intervals.

    Arguments
    ---------
    columns : sequence
        Sequence of indexes specifying the columns that have periodic data
    intervals : sequence of length-2 sequences
        Sequence of (min, max) intervals, one interval per column
    See help(cv2.kmeans) for all other arguments, which are passed through.

    Returns
    -------
    See help(cv2.kmeans) for outputs, which are passed through; except centers,
    which is modified so that it returns centers corresponding to the input
    data, instead of the transformed data.
    
    Raises
    ------
    cv2.error
        If len(columns) != len(intervals)
    """

    # Check each periodic column has an associated interval
    if len(columns) != len(intervals):
        raise cv2.error("number of intervals must be equal to number of columns")

    ndims = data.shape[1]
    ys = []

    # transform each periodic column into two columns with the x and y coordinate
    # of the angles for kmeans; x coord at original column, ys are appended
    for col, interval in zip(columns, intervals):
        a, b = min(interval), max(interval)
        width = b - a
        data[:, col] = TWO_PI * (data[:, col] - a) / width % TWO_PI
        ys.append(width * np.sin(data[:, col]))
        data[:, col] = width * np.cos(data[:, col])

    # append the ys to the end
    ys = np.array(ys).transpose()
    data = np.hstack((data, ys)).astype(np.float32)

    # run kmeans
    retval, bestLabels, centers = cv2.kmeans(data, *args, **kwargs)

    # transform the centers back to range they came from
    for i, (col, interval) in enumerate(zip(columns, intervals)):
        a, b = min(interval), max(interval)
        angles = np.arctan2(centers[:, ndims + i], centers[:, col]) % TWO_PI
        centers[:, col] = a + (b - a) * angles / TWO_PI
    centers = centers[:, :ndims]

    return retval, bestLabels, centers
