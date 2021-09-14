"""
Numba compiled codes for accelerating calculations
"""
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
