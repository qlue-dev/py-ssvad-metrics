from typing import List, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import auc, jaccard_score

from ssvad_metrics.data_schema import VADAnnotation, VADFrame


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


def traditional_criteria_masks(
        pred_masks: Union[np.ndarray, List[np.ndarray]],
        gt_masks: Union[np.ndarray, List[np.ndarray]]) -> dict:
    """
    Evaluate the single-scene video anomaly detection
    using the traditional criteria.

    PARAMETERS
    ----------
    pred_masks: Union[np.ndarray, List[np.ndarray]]
        Semantic mask video anomaly prediction results of all frames.
    gt_masks: Union[np.ndarray, List[np.ndarray]]
        Semantic mask video anomaly ground-truths of all frames.

    RETURN
    ------
    dict
        The results.
    """
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


def _get_rbdr_fpr_tbdr(
        pred_frms: List[VADFrame],
        gt_frms: List[VADFrame],
        threshold: float,
        alpha: float = 0.1,
        beta: float = 0.1) -> tuple:
    ntp, tar = 0, 0
    nfp, n_fs = 0, 0
    gt_a_trks, pred_a_trks = {}, {}
    for pred_f, gt_f in zip(pred_frms, gt_frms):
        track_id = gt_f.anomaly_track_id
        if track_id >= 0:
            # TODO: add behavior when no track_id (= None)
            # TODO: add behavior when no anomalous_regions (= None)
            gt_a_trk = gt_a_trks.setdefault(track_id, list())
            gt_a_trk.append(gt_f)
            pred_a_trk = pred_a_trks.setdefault(track_id, list())
            pred_a_trk.append(pred_f)
        n_fs += 1
        tar += len(gt_f.anomalous_regions)
        for gt_ar in gt_f.anomalous_regions:
            for pred_ar in pred_f.anomalous_regions:
                if pred_ar.score < threshold:
                    continue
                iou = calc_iou(gt_ar.bounding_box, pred_ar.bounding_box)
                if iou >= beta:
                    ntp += 1
                    break
        for pred_ar in pred_f.anomalous_regions:
            if pred_ar.score < threshold:
                continue
            for gt_ar in gt_f.anomalous_regions:
                iou = calc_iou(gt_ar.bounding_box, pred_ar.bounding_box)
                if iou >= beta:
                    break
            else:
                # pred_bbox do not overlap enough with any gt_bbox
                nfp += 1

    nat = len(gt_a_trks)

    ntpt = 0
    for gt_a_trk, pred_a_trk in zip(gt_a_trks.values(), pred_a_trks.values()):
        _tp = 0
        for gt_f, pred_f in zip(gt_a_trk, pred_a_trk):
            for gt_ar in gt_f.anomalous_regions:
                for pred_ar in pred_f.anomalous_regions:
                    if pred_ar.score < threshold:
                        continue
                    iou = calc_iou(gt_ar.bounding_box, pred_ar.bounding_box)
                    if iou >= beta:
                        _tp += 1
                        break
        if _tp >= (alpha * len(gt_a_trk)):
            ntpt += 1

    rbdr = ntp / tar
    fpr = nfp / n_fs
    tbdr = ntpt / nat
    return rbdr, fpr, tbdr


def current_criteria(
        preds: VADAnnotation,
        gts: VADAnnotation,
        alpha: float = 0.1,
        beta: float = 0.1) -> dict:
    anomaly_score_thresholds = np.linspace(1., 0., 1001)
    rbdrs, fprs, tbdrs = [], [], []
    for thr in anomaly_score_thresholds:
        rbdr, fpr, tbdr = _get_rbdr_fpr_tbdr(
            pred_frms=preds.frames,
            gt_frms=gts.frames,
            threshold=thr,
            alpha=alpha,
            beta=beta)
        rbdrs.append(rbdr)
        fprs.append(fpr)
        tbdrs.append(tbdr)
    result = {}
    result["track_roc_auc"] = auc(fprs, tbdrs)
    result["region_roc_auc"] = auc(fprs, rbdrs)
    return result
