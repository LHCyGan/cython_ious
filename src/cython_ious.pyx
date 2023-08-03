cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def bbox_iou(
        np.ndarray[DTYPE_t, ndim=2] boxes1,
        np.ndarray[DTYPE_t, ndim=2] boxes2):
    """
    Parameters
    ----------
    boxes1: (N, 4) ndarray of float
    boxes2: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes1 and boxes2
    """
    cdef unsigned int N = boxes1.shape[0]
    cdef unsigned int K = boxes2.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (boxes2[k, 2] - boxes2[k, 0] + 1) *
            (boxes2[k, 3] - boxes2[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes1[n, 2], boxes2[k, 2]) -
                max(boxes1[n, 0], boxes2[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes1[n, 3], boxes2[k, 3]) -
                    max(boxes1[n, 1], boxes2[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes1[n, 2] - boxes1[n, 0] + 1) *
                        (boxes1[n, 3] - boxes1[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def bbox_giou(
        np.ndarray[DTYPE_t, ndim=2] boxes1,
        np.ndarray[DTYPE_t, ndim=2] boxes2):
    """
    Parameters
    ----------
    boxes1: (N, 4) ndarray of float
    boxes2: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes1 and boxes2
    """
    cdef unsigned int N = boxes1.shape[0]
    cdef unsigned int K = boxes2.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area, area
    cdef DTYPE_t ua
    cdef unsigned int k, n
    cdef np.ndarray[DTYPE_t, ndim=2] lt, rb, wh

    for k in range(K):
        box_area = (
                (boxes2[k, 2] - boxes2[k, 0] + 1) *
                (boxes2[k, 3] - boxes2[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                    min(boxes1[n, 2], boxes2[k, 2]) -
                    max(boxes1[n, 0], boxes2[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                        min(boxes1[n, 3], boxes2[k, 3]) -
                        max(boxes1[n, 1], boxes2[k, 1]) + 1
                )

                if ih > 0:
                    ua = float(
                            (boxes1[n, 2] - boxes1[n, 0] + 1) *
                            (boxes1[n, 3] - boxes1[n, 1] + 1) +
                            box_area - iw * ih
                    )

                    lt = np.minimum(boxes1[n, :2], boxes2[k, :2])
                    rb = np.maximum(boxes1[n, 2:], boxes2[k, 2:])
                    wh = np.maximum(rb - lt, np.zeros_like(rb - lt))
                    area = wh[0] * wh[1]

                    overlaps[n, k] = iw * ih / ua
                    overlaps[n, k] = (overlaps[n, k] - (area - ua) / area)

    return overlaps


def bbox_diou(
        np.ndarray[DTYPE_t, ndim=2] boxes1,
        np.ndarray[DTYPE_t, ndim=2] boxes2):
    """
    Parameters
    ----------
    boxes1: (N, 4) ndarray of float
    boxes2: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes1 and boxes2
    """
    cdef unsigned int N = boxes1.shape[0]
    cdef unsigned int K = boxes2.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area, area
    cdef DTYPE_t ua, area_1, area_2
    cdef unsigned int k, n
    cdef DTYPE_t center_x1, center_y1, center_x2,  center_y2, p2

    for k in range(K):
        box_area = (
                (boxes2[k, 2] - boxes2[k, 0] + 1) *
                (boxes2[k, 3] - boxes2[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                    min(boxes1[n, 2], boxes2[k, 2]) -
                    max(boxes1[n, 0], boxes2[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                        min(boxes1[n, 3], boxes2[k, 3]) -
                        max(boxes1[n, 1], boxes2[k, 1]) + 1
                )

                if ih > 0:
                    ua = float(
                            (boxes1[n, 2] - boxes1[n, 0] + 1) *
                            (boxes1[n, 3] - boxes1[n, 1] + 1) +
                            box_area - iw * ih
                    )

                    area_1 = (boxes1[n, 2] - boxes1[n, 0]) * (boxes1[n, 3] - boxes1[n, 1])
                    area_2 = (boxes2[k, 2] - boxes2[k, 0]) * (boxes2[k, 3] - boxes2[k, 1])

                    # calculate center point of each box
                    center_x1 = (boxes1[n, 2] - boxes1[n, 0]) / 2
                    center_y1 = (boxes1[n, 3] - boxes1[n, 1]) / 2
                    center_x2 = (boxes2[k, 2] - boxes2[k, 0]) / 2
                    center_y2 = (boxes2[k, 3] - boxes2[k, 1]) / 2

                    # calculate square of center point distance
                    p2 = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2

                    # calculate square of the diagonal length
                    width_c = max(boxes1[n, 2], boxes2[k, 2]) - min(boxes1[n, 0], boxes2[k, 0])
                    height_c = max(boxes1[n, 3], boxes2[k, 3]) - min(boxes1[n, 1], boxes2[k, 1])
                    c2 = width_c ** 2 + height_c ** 2


                    overlaps[n, k] = iw * ih / ua
                    overlaps[n, k] = overlaps[n, k] - float(p2) / c2

    return overlaps