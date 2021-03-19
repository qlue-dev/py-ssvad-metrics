from typing import List, Union
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
from sklearn.metrics import auc


def _get_traditional_tpr_fpr(
        pred_masks: Union[np.ndarray, List[np.ndarray]],
        gt_masks: Union[np.ndarray, List[np.ndarray]],
        threshold: float) -> tuple:
    f_tp, f_fp, f_ps, f_ns = 0, 0, 0, 0
    p_tp, p_fp = 0, 0
    for pred_m, gt_m in zip(pred_masks, gt_masks):
        # GT
        gt_m_bool = gt_m.astype(np.bool)
        is_gt_m_pos = np.any(gt_m_bool)
        # Frame-level calc
        pred_m_pos = pred_m >= threshold
        f_is_pred_pos = np.any(pred_m_pos)
        # Counting f_tp, f_fp, f_ps, f_ns
        if is_gt_m_pos:
            f_ps += 1
            # Frame-level
            if f_is_pred_pos:
                f_tp += 1
            # Pixel-level
            p_is_pred_pos = np.sum(np.multiply(pred_m_pos, gt_m_bool)) \
                >= (0.4 * np.sum(gt_m_bool))
            if p_is_pred_pos:
                p_tp += 1
        else:
            f_ns += 1
            # Frame-level and pixel-level has same criterion
            if f_is_pred_pos:
                f_fp += 1
                p_fp += 1
    f_tpr_thr = f_tp / f_ps
    f_fpr_thr = f_fp / f_ns
    p_tpr_thr = p_tp / f_ps
    p_fpr_thr = p_fp / f_ns
    return f_tpr_thr, f_fpr_thr, p_tpr_thr, p_fpr_thr


def traditional_criteria(
        pred_masks: Union[np.ndarray, List[np.ndarray]],
        gt_masks: Union[np.ndarray, List[np.ndarray]]) -> dict:
    anomaly_score_thresholds = np.linspace(1., 0., 1001)
    f_tprs, f_fprs, p_tprs, p_fprs = [], [], [], []
    for thr in anomaly_score_thresholds:
        f_tpr_thr, f_fpr_thr, p_tpr_thr, p_fpr_thr = _get_traditional_tpr_fpr(
            pred_masks, gt_masks, thr)
        f_tprs.append(f_tpr_thr)
        f_fprs.append(f_fpr_thr)
        p_tprs.append(p_tpr_thr)
        p_fprs.append(p_fpr_thr)
    result = {}
    # Frame-level ROC AUC
    result["frame_roc_auc"] = auc(f_fprs, f_tprs)
    # Frame-level EER
    result["frame_eer"] = brentq(
        lambda x: 1. - x - interp1d(f_fprs, f_tprs)(x), 0., 1.)
    result["frame_thresh_at_eer"] = interp1d(
        f_fprs, anomaly_score_thresholds)(result["frame_eer"])
    # Pixel-level ROC AUC
    result["pixel_roc_auc"] = auc(p_fprs, p_tprs)
    # Pixel-level EER
    result["pixel_eer"] = brentq(
        lambda x: 1. - x - interp1d(p_fprs, p_tprs)(x), 0., 1.)
    result["pixel_thresh_at_eer"] = interp1d(
        p_fprs, anomaly_score_thresholds)(result["pixel_eer"])
    return result