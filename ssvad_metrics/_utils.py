"""
Numba compiled codes for accelerating calculations
"""
from typing import Tuple

import numpy as np
from numba import njit


@njit(fastmath=True)
def trad_pix_calc(pred_m_pos: np.ndarray, gt_m_bool: np.ndarray) -> float:
    return np.sum(np.multiply(pred_m_pos, gt_m_bool)) >= (0.4 * np.sum(gt_m_bool))


@njit(fastmath=True)
def mask_iou(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    Both array data must be boolean type.
    """
    intersection = np.logical_and(arr1, arr2)
    union = np.logical_or(arr1, arr2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


@njit(fastmath=True)
def bb_s_to_fp32_mask(
        bboxes: np.ndarray, scores: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create mask from bboxes and scores. On bboxes overlap, average their scores.
    """
    # create output mask
    accum_m = np.zeros(frame_shape, np.float32)
    ctr_m = np.zeros(frame_shape, np.float32)
    # draw bboxes on masks
    for i in range(len(bboxes)):
        bb = bboxes[i]
        ones_arr = np.ones_like(ctr_m[bb[1]:bb[3], bb[0]:bb[2]], np.float32)
        ctr_m[bb[1]:bb[3], bb[0]:bb[2]] += ones_arr
        accum_m[bb[1]:bb[3], bb[0]:bb[2]] += ones_arr * scores[i]
    return accum_m / ctr_m
