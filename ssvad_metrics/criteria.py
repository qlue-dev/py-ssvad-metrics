from typing import Tuple
import cv2
import numpy as np
from pydantic import BaseModel
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import auc

from ssvad_metrics._utils import mask_iou, trad_pix_calc
from ssvad_metrics.data_schema import VADAnnotation, VADFrame, load_pixel_score_map
from ssvad_metrics.utils import anomalous_regions_to_float_mask, connected_components

NUM_POINTS = 103
ANOMALY_SCORE_THRESHOLDS = np.linspace(1.01, -0.01, NUM_POINTS)


def _get_trad_calcs(
        preds: VADAnnotation, gts: VADAnnotation, threshold: float, use_region_mtrc: bool) -> tuple:
    f_tp, f_fp, f_ps, f_ns = 0, 0, 0, 0
    p_tp, p_fp = 0, 0
    if use_region_mtrc:
        pred_frm_shp = (preds.frame_height, preds.frame_width)
        gt_frm_shp = (gts.frame_height, gts.frame_width)
        if pred_frm_shp != gt_frm_shp:
            raise ValueError((
                "Predictions frame shape %s mismatched "
                "with the ground-truth %s!" % (pred_frm_shp, gt_frm_shp)))
        for pred_frm, gt_frm in zip(preds.frames, gts.frames):
            # GT
            if gt_frm.pixel_level_scores_map is not None:
                gt_m = load_pixel_score_map(gt_frm.pixel_level_scores_map)
                if gt_m.shape != gt_frm_shp:
                    raise ValueError((
                        "The loaded ground-truth anomaly score map frame shape %s "
                        "mismatched with the frame shape defined in the annotation %s!") % (gt_m.shape, gt_frm_shp))
            elif gt_frm.anomalous_regions is not None:
                gt_m = anomalous_regions_to_float_mask(
                    gt_frm.anomalous_regions, (gts.frame_height, gts.frame_width))
            else:
                raise ValueError((
                    "'is_anomalous_regions_available' is True "
                    "but no 'pixel_level_scores_map' nor 'anomalous_regions' "
                    "in frame_id=%d of GT file!") % gt_frm.frame_id)
            gt_m_bool = gt_m.astype(np.bool)
            is_gt_m_pos = np.any(gt_m_bool)
            # Frame-level calc
            if pred_frm.pixel_level_scores_map is not None:
                pred_m = load_pixel_score_map(pred_frm.pixel_level_scores_map)
                if pred_m.shape != pred_frm_shp:
                    raise ValueError((
                        "The loaded predictions anomaly score map frame shape %s "
                        "mismatched with the frame shape defined in the annotation %s!") % (pred_m.shape, pred_frm_shp))
            elif pred_frm.anomalous_regions is not None:
                pred_m = anomalous_regions_to_float_mask(
                    pred_frm.anomalous_regions, (preds.frame_height, preds.frame_width))
            else:
                raise ValueError((
                    "'is_anomalous_regions_available' is True "
                    "but no 'pixel_level_scores_map' nor 'anomalous_regions' "
                    "in frame_id=%d of pred file!") % pred_frm.frame_id)
            pred_m_pos = pred_m >= threshold
            f_is_pred_pos = np.any(pred_m_pos)
            # Counting f_tp, f_fp, f_ps, f_ns
            if is_gt_m_pos:
                f_ps += 1
                # Frame-level
                if f_is_pred_pos:
                    f_tp += 1
                # Pixel-level
                p_is_pred_pos = trad_pix_calc(pred_m_pos, gt_m_bool)
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
    return f_tp, f_fp, f_ps, f_ns, p_tp, p_fp


class _TradPerThrCalcsAccum(BaseModel):
    f_tp: float = 0
    f_fp: float = 0
    f_ps: float = 0
    f_ns: float = 0
    p_tp: float = 0
    p_fp: float = 0


class TraditionalCriteriaAccumulator:
    def __init__(self) -> None:
        """
        Accumulator for evaluating the single-scene video anomaly detection
        using the traditional criteria.
        Metrics are calculated from accumulated TPR and FPR.

        Reference: 
        B. Ramachandra, M. Jones and R. R. Vatsavai,
        "A Survey of Single-Scene Video Anomaly Detection,"
        in IEEE Transactions on Pattern Analysis and Machine Intelligence,
        doi: 10.1109/TPAMI.2020.3040591.
        """
        self.__result = {
            "frame_roc_auc": None,
            "frame_eer": None,
            "frame_thresh_at_eer": None,
            "pixel_roc_auc": None,
            "pixel_eer": None,
            "pixel_thresh_at_eer": None
        }
        self.__use_region_mtrc = []
        self.__per_thr_calcs_accum = {
            thr: _TradPerThrCalcsAccum()
            for thr in self.anomaly_score_thresholds
        }

    @property
    def anomaly_score_thresholds(self):
        return ANOMALY_SCORE_THRESHOLDS

    def update(
            self,
            preds: VADAnnotation,
            gts: VADAnnotation) -> None:
        """
        Update the accumulator with more prediction and groundtruth data pair.

        PARAMETERS
        ----------
        preds: VADAnnotation
            Video anomaly detection prediction result from a video.
        gts: VADAnnotation
            Video anomaly detection groundtruth of a video.
        """
        if preds.is_gt:
            raise ValueError(
                "The given prediction file has is_gt=True flag!")
        if gts.is_gt is not None and not gts.is_gt:
            raise ValueError(
                "The given ground-truth file has is_gt=False flag!")
        use_region_mtrc = preds.is_anomalous_regions_available and gts.is_anomalous_regions_available
        self.__use_region_mtrc.append(use_region_mtrc)
        for thr in self.anomaly_score_thresholds:
            f_tp, f_fp, f_ps, f_ns, p_tp, p_fp = _get_trad_calcs(
                preds, gts, thr, use_region_mtrc)
            _accum = self.__per_thr_calcs_accum[thr]
            _accum.f_tp += f_tp
            _accum.f_fp += f_fp
            _accum.f_ps += f_ps
            _accum.f_ns += f_ns
            _accum.p_tp += p_tp
            _accum.p_fp += p_fp

    def summarize(self) -> dict:
        """
        Summarize the accumulator.

        Pixel-level metrics only calculated if and only if all
        of the predictions and groundtruths are containing
        anomalous regions.

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
        if not self.__use_region_mtrc:
            # no any update
            return self.__result
        # calculate
        f_tprs, f_fprs, p_tprs, p_fprs = [], [], [], []
        for thr in self.anomaly_score_thresholds:
            _accum = self.__per_thr_calcs_accum[thr]
            f_tpr_thr = _accum.f_tp / _accum.f_ps
            f_fpr_thr = _accum.f_fp / _accum.f_ns
            if all(self.__use_region_mtrc):
                p_tpr_thr = _accum.p_tp / _accum.f_ps
                p_fpr_thr = _accum.p_fp / _accum.f_ns
            else:
                p_tpr_thr = None
                p_fpr_thr = None
            f_tprs.append(f_tpr_thr)
            f_fprs.append(f_fpr_thr)
            p_tprs.append(p_tpr_thr)
            p_fprs.append(p_fpr_thr)
        # Frame-level ROC AUC
        f_sorted_indices = np.argsort(f_fprs)
        f_fprs = [
            f_fprs[i] for i in f_sorted_indices]
        f_tprs = [
            f_tprs[i] for i in f_sorted_indices]
        self.__result["frame_roc_auc"] = auc(f_fprs, f_tprs)
        # Frame-level EER
        self.__result["frame_eer"] = brentq(
            lambda x: 1. - x - interp1d(f_fprs, f_tprs)(x), 0., 1.)
        self.__result["frame_thresh_at_eer"] = float(interp1d(
            f_fprs, self.anomaly_score_thresholds)(self.__result["frame_eer"]))
        if all(self.__use_region_mtrc):
            # sort x axis
            p_sorted_indices = np.argsort(p_fprs)
            p_fprs = [
                p_fprs[i] for i in p_sorted_indices]
            p_tprs = [
                p_tprs[i] for i in p_sorted_indices]
            # Pixel-level ROC AUC
            self.__result["pixel_roc_auc"] = auc(p_fprs, p_tprs)
            # Pixel-level EER
            self.__result["pixel_eer"] = brentq(
                lambda x: 1. - x - interp1d(p_fprs, p_tprs)(x), 0., 1.)
            self.__result["pixel_thresh_at_eer"] = float(interp1d(
                p_fprs, self.anomaly_score_thresholds)(self.__result["pixel_eer"]))
        else:
            self.__result["pixel_roc_auc"] = None
            self.__result["pixel_eer"] = None
            self.__result["pixel_thresh_at_eer"] = None
        return self.__result

    def __call__(
            self,
            preds: VADAnnotation,
            gts: VADAnnotation) -> dict:
        """
        Update the accumulator with data and summarize the accumulator.
        Useful when there are only single prediction and groundtruth data pair.

        Pixel-level metrics only calculated if and only if all
        of the predictions and groundtruths are containing
        anomalous regions.

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
        self.update(preds=preds, gts=gts)
        return self.summarize()


def _get_connected_components(
        threshold: float,
        gt_frm: VADFrame,
        gt_frm_shape: Tuple[int, int],
        pred_frm: VADFrame,
        pred_frm_shape: Tuple[int, int]) -> Tuple[int, np.ndarray, int, np.ndarray]:
    if gt_frm.pixel_level_scores_map is not None:
        gt_m = load_pixel_score_map(gt_frm.pixel_level_scores_map)
    elif gt_frm.anomalous_regions is not None:
        gt_m = anomalous_regions_to_float_mask(
            gt_frm.anomalous_regions, gt_frm_shape)
    else:
        raise ValueError((
            "'is_anomalous_regions_available' is True "
            "but no 'pixel_level_scores_map' nor 'anomalous_regions' "
            "in frame_id=%d of GT file!") % gt_frm.frame_id)
    gt_m = gt_m.astype(np.bool) * np.uint8(255)
    gt_num_ccs, gt_cc_labels = connected_components(gt_m)

    if pred_frm.pixel_level_scores_map is not None:
        pred_m = load_pixel_score_map(pred_frm.pixel_level_scores_map)
    elif pred_frm.anomalous_regions is not None:
        pred_m = anomalous_regions_to_float_mask(
            pred_frm.anomalous_regions, pred_frm_shape)
    else:
        raise ValueError((
            "'is_anomalous_regions_available' is True "
            "but no 'pixel_level_scores_map' nor 'anomalous_regions' "
            "in frame_id=%d of pred file!") % pred_frm.frame_id)
    pred_m = (pred_m >= threshold) * np.uint8(255)
    pred_num_ccs, pred_cc_labels = connected_components(pred_m)

    return gt_num_ccs, gt_cc_labels, pred_num_ccs, pred_cc_labels


def _get_cur_calcs(
        preds: VADAnnotation,
        gts: VADAnnotation,
        threshold: float,
        alpha: float,
        beta: float,
        use_region_mtrc: bool,
        use_track_mtrc: bool) -> tuple:
    pred_frms = preds.frames
    gt_frms = gts.frames
    ntp, tar = 0, 0
    nfp, n_fs = 0, 0
    gt_a_trks, pred_a_trks = {}, {}
    # calculate frame-by-frame
    for pred_f, gt_f in zip(pred_frms, gt_frms):
        n_fs += 1
        if use_track_mtrc:
            # check for track_id for track metric calculation
            track_id = gt_f.anomaly_track_id
            if track_id >= 0:
                gt_a_trk = gt_a_trks.setdefault(track_id, list())
                gt_a_trk.append(gt_f)
                pred_a_trk = pred_a_trks.setdefault(track_id, list())
                pred_a_trk.append(pred_f)
        if use_region_mtrc:
            # load mask and get connected components
            # NOTE: number of connected components always include backgroud
            # (even all pixels are True). label 0 is the background (BG).
            gt_num_ccs, gt_cc_labels, pred_num_ccs, pred_cc_labels = _get_connected_components(
                threshold,
                gt_f, (gts.frame_height, gts.frame_width),
                pred_f, (preds.frame_height, preds.frame_width))
            # calculate region metric
            tar += gt_num_ccs - 1  # minus BG
            for k in range(1, gt_num_ccs):  # skip BG
                gt_ar_m = gt_cc_labels == k
                for j in range(1, pred_num_ccs):  # skip BG
                    pred_ar_m = pred_cc_labels == j
                    iou = mask_iou(gt_ar_m, pred_ar_m)
                    if iou >= beta:
                        ntp += 1
                        break
            for j in range(1, pred_num_ccs):
                pred_ar_m = pred_cc_labels == j
                for k in range(1, gt_num_ccs):
                    gt_ar_m = gt_cc_labels == k
                    iou = mask_iou(gt_ar_m, pred_ar_m)
                    if iou >= beta:
                        break
                else:
                    # pred_ar_m do not IoU enough with any gt_ar_m
                    nfp += 1

    nat = len(gt_a_trks)
    ntpt = 0
    if use_region_mtrc and use_track_mtrc:
        for gt_a_trk, pred_a_trk in zip(gt_a_trks.values(), pred_a_trks.values()):
            _tp = 0
            lk_size = 0
            for gt_f, pred_f in zip(gt_a_trk, pred_a_trk):
                gt_num_ccs, gt_cc_labels, pred_num_ccs, pred_cc_labels = _get_connected_components(
                    threshold,
                    gt_f, (gts.frame_height, gts.frame_width),
                    pred_f, (preds.frame_height, preds.frame_width))
                for k in range(1, gt_num_ccs):  # skip BG
                    gt_ar_m = gt_cc_labels == k
                    lk_size += 1
                    for j in range(1, pred_num_ccs):  # skip BG
                        pred_ar_m = pred_cc_labels == j
                        iou = mask_iou(gt_ar_m, pred_ar_m)
                        if iou >= beta:
                            _tp += 1
                            break
            if _tp >= (alpha * lk_size):
                ntpt += 1
    return ntp, tar, nfp, n_fs, nat, ntpt


class _CurPerThrCalcsAccum(BaseModel):
    ntp: float = 0
    tar: float = 0
    nfp: float = 0
    n_fs: float = 0
    nat: float = 0
    ntpt: float = 0


class CurrentCriteriaAccumulator:
    def __init__(
            self,
            alpha: float = 0.1,
            beta: float = 0.1) -> None:
        """
        Accumulator for evaluating the single-scene video anomaly detection
        using the "current" criteria.
        Metrics are calculated from accumulated RBDR, TBDR, and FPR.

        Metrics can be only calculated if and only if all
        of the predictions and groundtruths are containing
        anomalous regions.

        Reference: 
        B. Ramachandra, M. Jones and R. R. Vatsavai,
        "A Survey of Single-Scene Video Anomaly Detection,"
        in IEEE Transactions on Pattern Analysis and Machine Intelligence,
        doi: 10.1109/TPAMI.2020.3040591.

        PARAMETERS
        ----------
        alpha: float = 0.1
            A threshold used in NTPT calculation. See reference for more information.
        beta: float = 0.1
            A threshold used in NTP, NFP, and NTPT calculations. See reference for more information.
        """
        self.__alpha = alpha
        self.__beta = beta
        self.rbdrs, self.fprs, self.tbdrs = [], [], []
        self.__result = {
            "region_roc_auc": None,
            "track_roc_auc": None
        }
        self.__use_region_mtrc = []
        self.__use_track_mtrc = []
        self.__per_thr_calcs_accum = {
            thr: _CurPerThrCalcsAccum()
            for thr in self.anomaly_score_thresholds
        }

    @property
    def anomaly_score_thresholds(self):
        return ANOMALY_SCORE_THRESHOLDS

    @property
    def alpha(self) -> float:
        return self.__alpha

    @property
    def beta(self) -> float:
        return self.__beta

    def update(
            self,
            preds: VADAnnotation,
            gts: VADAnnotation) -> None:
        """
        Update the accumulator with more prediction and groundtruth data pair.

        PARAMETERS
        ----------
        preds: VADAnnotation
            Video anomaly detection prediction result from a video.
        gts: VADAnnotation
            Video anomaly detection groundtruth of a video.
        """
        if preds.is_gt:
            raise ValueError(
                "The given prediction file has is_gt=True flag!")
        if gts.is_gt is not None and not gts.is_gt:
            raise ValueError(
                "The given ground-truth file has is_gt=False flag!")
        use_region_mtrc = preds.is_anomalous_regions_available and gts.is_anomalous_regions_available
        self.__use_region_mtrc.append(use_region_mtrc)
        use_track_mtrc = gts.is_anomaly_track_id_available
        self.__use_track_mtrc.append(use_track_mtrc)
        for thr in self.anomaly_score_thresholds:
            ntp, tar, nfp, n_fs, nat, ntpt = _get_cur_calcs(
                preds, gts, thr, self.alpha, self.beta, use_region_mtrc, use_track_mtrc)
            _accum = self.__per_thr_calcs_accum[thr]
            _accum.ntp += ntp
            _accum.tar += tar
            _accum.nfp += nfp
            _accum.n_fs += n_fs
            _accum.nat += nat
            _accum.ntpt += ntpt

    def summarize(self) -> dict:
        """
        Summarize the accumulator.

        Track-based metrics can be only calculated only if all
        of groundtruths are containing anomalous tracks. 
        Anomalous tracks not required in predictions.

        RETURN
        ------
        Dict[str, Any]
            Calculated performance metrics: "region_roc_auc" and "track_roc_auc".
        """
        if not (self.__use_region_mtrc and all(self.__use_region_mtrc)):
            return {
                "region_roc_auc": None,
                "track_roc_auc": None
            }
        # calculate
        rbdrs, fprs, tbdrs = [], [], []
        for thr in self.anomaly_score_thresholds:
            _accum = self.__per_thr_calcs_accum[thr]
            rbdr = _accum.ntp / _accum.tar
            fpr = _accum.nfp / _accum.n_fs
            if all(self.__use_track_mtrc):
                tbdr = _accum.ntpt / _accum.nat
            else:
                tbdr = None
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
        sorted_indices = np.argsort(fprs_)
        fprs_ = [
            fprs_[i] for i in sorted_indices]
        rbdrs_ = [
            rbdrs_[i] for i in sorted_indices]
        tbdrs_ = [
            tbdrs_[i] for i in sorted_indices]
        self.__result["region_roc_auc"] = auc(fprs_, rbdrs_)
        if all(self.__use_track_mtrc):
            self.__result["track_roc_auc"] = auc(fprs_, tbdrs_)
        else:
            self.__result["track_roc_auc"] = None
        return self.__result

    def __call__(
            self,
            preds: VADAnnotation,
            gts: VADAnnotation) -> dict:
        """
        Update the accumulator with data and summarize the accumulator.
        Useful when there are only single prediction and groundtruth data pair.

        Pixel-level metrics only calculated if and only if all
        of the predictions and groundtruths are containing
        anomalous regions.

        PARAMETERS
        ----------
        preds: VADAnnotation
            Video anomaly detection prediction result from a video.
        gts: VADAnnotation
            Video anomaly detection groundtruth of a video.

        RETURN
        ------
        Dict[str, Any]
            Calculated performance metrics: "region_roc_auc" and "track_roc_auc".
        """
        self.update(preds=preds, gts=gts)
        return self.summarize()
