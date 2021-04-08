from typing import Iterable, Tuple

import cv2
import numpy as np


def bounding_boxes_to_float_mask(
        bboxes: Iterable, scores: Iterable, frame_shape: Tuple[int]) -> np.ndarray:
    """
    Convert bounding boxes into a semantic mask with floating value.
    """
    _m = np.zeros(frame_shape, np.float32)
    for bbox, score in zip(bboxes, scores):
        cv2.rectangle(_m, bbox, color=score, thickness=-1)
    return _m


def anomalous_regions_to_float_mask(
        anomalous_regions: list, frame_shape: Tuple[int]) -> np.ndarray:
    """
    Convert list of `AnomalousRegion` instances into a semantic mask with floating value.
    """
    _m = np.zeros(frame_shape, np.float32)
    for _ar in anomalous_regions:
        cv2.rectangle(_m, _ar.bounding_box, color=_ar.score, thickness=-1)
    return _m


def iou_single(bb1: Iterable, bb2: Iterable) -> float:
    """
    Computes intersection over union.

    Parameters
    ----------
    bb1 : Iterable
        A bounding box in format (top left x, top left y, bottom right x, bottom right y).
    bb2 : Iterable
        A bounding box in format (top left x, top left y, bottom right x, bottom right y).

    Returns
    -------
    float
        The intersection over union in [0, 1] between the `bb1` and `bb2`.

    """
    bb1 = np.asarray(bb1)
    bb2 = np.asarray(bb2)
    intersections_tl = np.r_[
        np.maximum(bb1[0], bb2[0]), np.maximum(bb1[1], bb2[1])]
    intersections_br = np.r_[
        np.minimum(bb1[2], bb2[2]), np.minimum(bb1[3], bb2[3])]
    intersections_wh = np.maximum(0., intersections_br - intersections_tl)
    intersections_area = intersections_wh.prod()
    bb1_area = (bb1[2:] - bb1[:2]).prod()
    bb2_area = (bb2[2:] - bb2[:2]).prod()
    return intersections_area / (bb1_area + bb2_area - intersections_area)


def iou(box, candidates):
    """
    Computes intersection over union.

    Parameters
    ----------
    box : ndarray
        A bounding box in format (top left x, top left y, bottom right x, bottom right y).
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `box`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `box` and each
        candidate. A higher score means a larger fraction of the `box` is
        occluded by the candidate.

    """
    intersections_tl = np.c_[
        np.maximum(box[0], candidates[:, 0])[:, np.newaxis],
        np.maximum(box[1], candidates[:, 1])[:, np.newaxis]]
    intersections_br = np.c_[
        np.minimum(box[2], candidates[:, 2])[:, np.newaxis],
        np.minimum(box[3], candidates[:, 3])[:, np.newaxis]]
    intersections_wh = np.maximum(0., intersections_br - intersections_tl)
    intersections_area = intersections_wh.prod(axis=1)
    box_area = (np.asarray(box)[2:] - np.asarray(box)[: 2]).prod()
    candidates_area = (candidates[:, 2:] - candidates[:, : 2]).prod(axis=1)
    return intersections_area / (box_area + candidates_area - intersections_area)
