from typing import List, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import auc, jaccard_score

from ssvad_metrics.data_schema import VADAnnotation
from ssvad_metrics.utils import anomalous_regions_to_float_mask, iou_single

NUM_POINTS = 103

# def _get_traditional_tpr_fpr_masks(
#         pred_masks: Union[np.ndarray, List[np.ndarray]],
#         gt_masks: Union[np.ndarray, List[np.ndarray]],
#         threshold: float) -> tuple:
#     f_tp, f_fp, f_ps, f_ns = 0, 0, 0, 0
#     p_tp, p_fp = 0, 0
#     for pred_m, gt_m in zip(pred_masks, gt_masks):
#         # GT
#         gt_m_bool = gt_m.astype(np.bool)
#         is_gt_m_pos = np.any(gt_m_bool)
#         # Frame-level calc
#         pred_m_pos = pred_m >= threshold
#         f_is_pred_pos = np.any(pred_m_pos)
#         # Counting f_tp, f_fp, f_ps, f_ns
#         if is_gt_m_pos:
#             f_ps += 1
#             # Frame-level
#             if f_is_pred_pos:
#                 f_tp += 1
#             # Pixel-level
#             p_is_pred_pos = np.sum(np.multiply(pred_m_pos, gt_m_bool)) \
#                 >= (0.4 * np.sum(gt_m_bool))
#             if p_is_pred_pos:
#                 p_tp += 1
#         else:
#             f_ns += 1
#             # Frame-level and pixel-level has same criterion
#             if f_is_pred_pos:
#                 f_fp += 1
#                 p_fp += 1
#     f_tpr_thr = f_tp / f_ps
#     f_fpr_thr = f_fp / f_ns
#     p_tpr_thr = p_tp / f_ps
#     p_fpr_thr = p_fp / f_ns
#     return f_tpr_thr, f_fpr_thr, p_tpr_thr, p_fpr_thr


# def traditional_criteria_masks(
#         pred_masks: Union[np.ndarray, List[np.ndarray]],
#         gt_masks: Union[np.ndarray, List[np.ndarray]]) -> dict:
#     """
#     Evaluate the single-scene video anomaly detection
#     using the traditional criteria.

#     Reference:
#     B. Ramachandra, M. Jones and R. R. Vatsavai,
#     "A Survey of Single-Scene Video Anomaly Detection,"
#     in IEEE Transactions on Pattern Analysis and Machine Intelligence,
#     doi: 10.1109/TPAMI.2020.3040591.

#     PARAMETERS
#     ----------
#     pred_masks: Union[np.ndarray, List[np.ndarray]]
#         Semantic mask video anomaly prediction results of all frames.
#     gt_masks: Union[np.ndarray, List[np.ndarray]]
#         Semantic mask video anomaly ground-truths of all frames.

#     RETURN
#     ------
#     dict
#         The results.
#     """
#     anomaly_score_thresholds = np.linspace(1., 0., 1001)
#     f_tprs, f_fprs, p_tprs, p_fprs = [], [], [], []
#     for thr in anomaly_score_thresholds:
#         f_tpr_thr, f_fpr_thr, p_tpr_thr, p_fpr_thr = _get_traditional_tpr_fpr_masks(
#             pred_masks, gt_masks, thr)
#         f_tprs.append(f_tpr_thr)
#         f_fprs.append(f_fpr_thr)
#         p_tprs.append(p_tpr_thr)
#         p_fprs.append(p_fpr_thr)
#     result = {}
#     # Frame-level ROC AUC
#     result["frame_roc_auc"] = auc(f_fprs, f_tprs)
#     # Frame-level EER
#     result["frame_eer"] = brentq(
#         lambda x: 1. - x - interp1d(f_fprs, f_tprs)(x), 0., 1.)
#     result["frame_thresh_at_eer"] = interp1d(
#         f_fprs, anomaly_score_thresholds)(result["frame_eer"])
#     # Pixel-level ROC AUC
#     result["pixel_roc_auc"] = auc(p_fprs, p_tprs)
#     # Pixel-level EER
#     result["pixel_eer"] = brentq(
#         lambda x: 1. - x - interp1d(p_fprs, p_tprs)(x), 0., 1.)
#     result["pixel_thresh_at_eer"] = interp1d(
#         p_fprs, anomaly_score_thresholds)(result["pixel_eer"])
#     return result


def _get_traditional_tpr_fpr(
        preds: VADAnnotation,
        gts: VADAnnotation,
        threshold: float) -> tuple:
    use_region_mtrc = preds.is_anomalous_regions_available and gts.is_anomalous_regions_available
    f_tp, f_fp, f_ps, f_ns = 0, 0, 0, 0
    p_tp, p_fp = 0, 0
    if use_region_mtrc:
        pred_frm_shp = (preds.frame_height, preds.frame_width)
        gt_frm_shp = (gts.frame_height, gts.frame_width)
        for pred_frm, gt_frm in zip(preds.frames, gts.frames):
            # GT
            gt_m = anomalous_regions_to_float_mask(
                gt_frm.anomalous_regions, gt_frm_shp)
            gt_m_bool = gt_m.astype(np.bool)
            is_gt_m_pos = np.any(gt_m_bool)
            # Frame-level calc
            pred_m = anomalous_regions_to_float_mask(
                pred_frm.anomalous_regions, pred_frm_shp)
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
    else:
        for pred_frm, gt_frm in zip(preds.frames, gts.frames):
            # GT
            is_gt_m_pos = bool(gt_frm.frame_level_score)
            # Frame-level calc
            f_is_pred_pos = pred_frm.frame_level_score >= threshold
            # Counting f_tp, f_fp, f_ps, f_ns
            if is_gt_m_pos:
                f_ps += 1
                # Frame-level
                if f_is_pred_pos:
                    f_tp += 1
            else:
                f_ns += 1
                # Frame-level
                if f_is_pred_pos:
                    f_fp += 1
    f_tpr_thr = f_tp / f_ps
    f_fpr_thr = f_fp / f_ns
    if use_region_mtrc:
        p_tpr_thr = p_tp / f_ps
        p_fpr_thr = p_fp / f_ns
    else:
        p_tpr_thr = None
        p_fpr_thr = None
    return f_tpr_thr, f_fpr_thr, p_tpr_thr, p_fpr_thr


def traditional_criteria(
        preds: VADAnnotation,
        gts: VADAnnotation) -> dict:
    """
    Evaluate the single-scene video anomaly detection
    using the traditional criteria.

    Reference: 
    B. Ramachandra, M. Jones and R. R. Vatsavai,
    "A Survey of Single-Scene Video Anomaly Detection,"
    in IEEE Transactions on Pattern Analysis and Machine Intelligence,
    doi: 10.1109/TPAMI.2020.3040591.

    PARAMETERS
    ----------
    preds: VADAnnotation
        Video anomaly detection prediction result from a video.
    gts: VADAnnotation
        Video anomaly detection groundtruth of a video.

    RETURN
    ------
    Dict[str, Any]
        Calculated performance metrics: 
        "frame_roc_auc",
        "frame_eer",
        "frame_thresh_at_eer",
        "pixel_roc_auc",
        "pixel_eer", and
        "pixel_thresh_at_eer".
    """
    result = {
        "frame_roc_auc": None,
        "frame_eer": None,
        "frame_thresh_at_eer": None,
        "pixel_roc_auc": None,
        "pixel_eer": None,
        "pixel_thresh_at_eer": None
    }
    use_region_mtrc = preds.is_anomalous_regions_available and gts.is_anomalous_regions_available
    anomaly_score_thresholds = np.linspace(1.01, -0.01, NUM_POINTS)
    f_tprs, f_fprs, p_tprs, p_fprs = [], [], [], []
    for thr in anomaly_score_thresholds:
        f_tpr_thr, f_fpr_thr, p_tpr_thr, p_fpr_thr = _get_traditional_tpr_fpr(
            preds, gts, thr)
        f_tprs.append(f_tpr_thr)
        f_fprs.append(f_fpr_thr)
        p_tprs.append(p_tpr_thr)
        p_fprs.append(p_fpr_thr)
    # Frame-level ROC AUC
    result["frame_roc_auc"] = auc(f_fprs, f_tprs)
    # Frame-level EER
    result["frame_eer"] = brentq(
        lambda x: 1. - x - interp1d(f_fprs, f_tprs)(x), 0., 1.)
    result["frame_thresh_at_eer"] = float(interp1d(
        f_fprs, anomaly_score_thresholds)(result["frame_eer"]))
    if use_region_mtrc:
        # Pixel-level ROC AUC
        result["pixel_roc_auc"] = auc(p_fprs, p_tprs)
        # Pixel-level EER
        result["pixel_eer"] = brentq(
            lambda x: 1. - x - interp1d(p_fprs, p_tprs)(x), 0., 1.)
        result["pixel_thresh_at_eer"] = float(interp1d(
            p_fprs, anomaly_score_thresholds)(result["pixel_eer"]))
    return result


def _get_rbdr_fpr_tbdr(
        preds: VADAnnotation,
        gts: VADAnnotation,
        threshold: float,
        alpha: float = 0.1,
        beta: float = 0.1) -> tuple:
    pred_frms = preds.frames
    gt_frms = gts.frames
    use_region_mtrc = preds.is_anomalous_regions_available and gts.is_anomalous_regions_available
    use_track_mtrc = preds.is_anomaly_track_id_available and gts.is_anomaly_track_id_available
    ntp, tar = 0, 0
    nfp, n_fs = 0, 0
    gt_a_trks, pred_a_trks = {}, {}
    for pred_f, gt_f in zip(pred_frms, gt_frms):
        n_fs += 1
        if use_track_mtrc:
            track_id = gt_f.anomaly_track_id
            if track_id >= 0:
                gt_a_trk = gt_a_trks.setdefault(track_id, list())
                gt_a_trk.append(gt_f)
                pred_a_trk = pred_a_trks.setdefault(track_id, list())
                pred_a_trk.append(pred_f)
        if use_region_mtrc:
            tar += len(gt_f.anomalous_regions)
            for gt_ar in gt_f.anomalous_regions:
                for pred_ar in pred_f.anomalous_regions:
                    if pred_ar.score < threshold:
                        continue
                    iou = iou_single(gt_ar.bounding_box, pred_ar.bounding_box)
                    if iou >= beta:
                        ntp += 1
                        break
            for pred_ar in pred_f.anomalous_regions:
                if pred_ar.score < threshold:
                    continue
                for gt_ar in gt_f.anomalous_regions:
                    iou = iou_single(gt_ar.bounding_box, pred_ar.bounding_box)
                    if iou >= beta:
                        break
                else:
                    # pred_bbox do not overlap enough with any gt_bbox
                    nfp += 1

    if use_track_mtrc:
        nat = len(gt_a_trks)
        ntpt = 0
        for gt_a_trk, pred_a_trk in zip(gt_a_trks.values(), pred_a_trks.values()):
            _tp = 0
            for gt_f, pred_f in zip(gt_a_trk, pred_a_trk):
                for gt_ar in gt_f.anomalous_regions:
                    for pred_ar in pred_f.anomalous_regions:
                        if pred_ar.score < threshold:
                            continue
                        iou = iou_single(gt_ar.bounding_box,
                                         pred_ar.bounding_box)
                        if iou >= beta:
                            _tp += 1
                            break
            if _tp >= (alpha * len(gt_a_trk)):
                ntpt += 1
    if use_region_mtrc:
        rbdr = ntp / tar
        fpr = nfp / n_fs
    else:
        rbdr = None
        fpr = None
    if use_region_mtrc and use_track_mtrc:
        tbdr = ntpt / nat
    else:
        tbdr = None
    return rbdr, fpr, tbdr


def current_criteria(
        preds: VADAnnotation,
        gts: VADAnnotation,
        alpha: float = 0.1,
        beta: float = 0.1) -> dict:
    """
    Evaluate the single-scene video anomaly detection
    using the "current" criteria.

    Reference: 
    B. Ramachandra, M. Jones and R. R. Vatsavai,
    "A Survey of Single-Scene Video Anomaly Detection,"
    in IEEE Transactions on Pattern Analysis and Machine Intelligence,
    doi: 10.1109/TPAMI.2020.3040591.

    PARAMETERS
    ----------
    preds: VADAnnotation
        Video anomaly detection prediction result from a video.
    gts: VADAnnotation
        Video anomaly detection groundtruth of a video.
    alpha: float = 0.1
        A threshold used in NTPT calculation. See reference for more information.
    beta: float = 0.1
        A threshold used in NTP, NFP, and NTPT calculations. See reference for more information.

    RETURN
    ------
    Dict[str, Any]
        Calculated performance metrics: "region_roc_auc" and "track_roc_auc".
    """
    result = {
        "region_roc_auc": None,
        "track_roc_auc": None
    }
    use_region_mtrc = preds.is_anomalous_regions_available and gts.is_anomalous_regions_available
    use_track_mtrc = preds.is_anomaly_track_id_available and gts.is_anomaly_track_id_available
    if use_region_mtrc:
        anomaly_score_thresholds = np.linspace(1.01, -0.01, NUM_POINTS)
        rbdrs, fprs, tbdrs = [], [], []
        for thr in anomaly_score_thresholds:
            rbdr, fpr, tbdr = _get_rbdr_fpr_tbdr(
                preds=preds,
                gts=gts,
                threshold=thr,
                alpha=alpha,
                beta=beta)
            rbdrs.append(rbdr)
            fprs.append(fpr)
            tbdrs.append(tbdr)
        # recommendation calculating AUC for false positive rates per frame from 0 to 1.0
        rbdrs_, fprs_, tbdrs_ = [], [], []
        for rbdr, fpr, tbdr in zip(rbdrs, fprs, tbdrs):
            if not (0.0 <= fpr <= 1.0):
                continue
            rbdrs_.append(rbdr)
            fprs_.append(fpr)
            tbdrs_.append(tbdr)
        result["region_roc_auc"] = auc(fprs_, rbdrs_)
        if use_track_mtrc:
            result["track_roc_auc"] = auc(fprs_, tbdrs_)
    return result
