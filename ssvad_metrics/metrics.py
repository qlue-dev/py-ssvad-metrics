import json
import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ssvad_metrics import data_schema
from ssvad_metrics.criteria import (CurrentCriteriaAccumulator,
                                    TraditionalCriteriaAccumulator)

logger = logging.getLogger()


def evaluate(
        gt_path: str,
        pred_path: str,
        gt_score_maps_root_dir: Optional[str] = None,
        pred_score_maps_root_dir: Optional[str] = None,
        alpha: float = 0.1,
        beta: float = 0.1) -> dict:
    """
    Evaluate the single-scene video anomaly detection
    using the traditional criteria, and
    using the "current" criteria.

    Reference:
    B. Ramachandra, M. Jones and R. R. Vatsavai,
    "A Survey of Single-Scene Video Anomaly Detection,"
    in IEEE Transactions on Pattern Analysis and Machine Intelligence,
    doi: 10.1109/TPAMI.2020.3040591.


    PARAMETERS
    ----------
    gt_path: str
        Path to VADAnnotation-formatted JSON file containing the ground truth annotation
        of the video anomaly detection. See `data_schema.VADAnnotation.schema()` or
        `data_schema.VADAnnotation.schema_json()` for the JSON schema.
    pred_path: str
        Path to VADAnnotation-formatted JSON file containing the prediction results
        of the video anomaly detection. See `data_schema.VADAnnotation.schema()` or
        `data_schema.VADAnnotation.schema_json()` for the JSON schema.
    gt_score_maps_root_dir: Optional[str] = None
        The root directory for the pixel-level anomaly scores maps files in the ground-truth JSON.
    pred_score_maps_root_dir: Optional[str] = None
        The root directory for the pixel-level anomaly scores maps files in the prediction JSON.
    alpha: float = 0.1
        A threshold used in NTPT calculation. See reference for more information.
    beta: float = 0.1
        A threshold used in NTP, NFP, and NTPT calculations. See reference for more information.

    RETURN
    ------
    Dict[str, Any]
    """
    with open(gt_path, "r") as fp:
        gt_annos = data_schema.data_parser(
            json.load(fp), gt_score_maps_root_dir)
    with open(pred_path, "r") as fp:
        pred_annos = data_schema.data_parser(
            json.load(fp), pred_score_maps_root_dir)
    if gt_annos.frames_count != pred_annos.frames_count:
        raise ValueError("Frames count Pred != frames count GT")
    results = {}
    results.update(
        TraditionalCriteriaAccumulator()(preds=pred_annos, gts=gt_annos)
    )
    results.update(
        CurrentCriteriaAccumulator(alpha=alpha, beta=beta)(preds=pred_annos, gts=gt_annos))
    return results


def accumulated_evaluate(
        gt_dir: str,
        pred_dir: str,
        gt_name_suffix: str = "",
        pred_name_suffix: str = "",
        gt_score_maps_root_dir: Optional[str] = None,
        pred_score_maps_root_dir: Optional[str] = None,
        alpha: float = 0.1,
        beta: float = 0.1,
        show_progress: bool = True) -> dict:
    """
    Evaluate the single-scene video anomaly detection
    using the traditional criteria, and using the "current" criteria.

    Useful when the single-scene video is splitted into multiple clips
    and predicted by a single model with a single configuration,
    hence there are multiple JSON files, but single
    aggregated/accumulated calculation is required.
    Both directory must contain exactly same files count and file names
    (excluding suffix and file extension).

    Reference:
    B. Ramachandra, M. Jones and R. R. Vatsavai,
    "A Survey of Single-Scene Video Anomaly Detection,"
    in IEEE Transactions on Pattern Analysis and Machine Intelligence,
    doi: 10.1109/TPAMI.2020.3040591.


    PARAMETERS
    ----------
    gt_dir: str
        Path to directory containing VADAnnotation-formatted JSON files.
        Each JSON file containing the ground truth annotation
        of the video anomaly detection. See `data_schema.VADAnnotation.schema()` or
        `data_schema.VADAnnotation.schema_json()` for the JSON schema.
    pred_dir: str
        Path to directory containing VADAnnotation-formatted JSON files.
        Each JSON file containing the prediction results
        of the video anomaly detection. See `data_schema.VADAnnotation.schema()` or
        `data_schema.VADAnnotation.schema_json()` for the JSON schema.
    gt_name_suffix: str = ""
        Fixed file name suffix, if any. Do not include the file extension.
    pred_name_suffix: str = ""
        Fixed file name suffix, if any. Do not include the file extension.
    gt_score_maps_root_dir: Optional[str] = None
        The root directory for the pixel-level anomaly scores maps files in the ground-truth JSON.
    pred_score_maps_root_dir: Optional[str] = None
        The root directory for the pixel-level anomaly scores maps files in the prediction JSON.
    alpha: float = 0.1
        A threshold used in NTPT calculation. See reference for more information.
    beta: float = 0.1
        A threshold used in NTP, NFP, and NTPT calculations. See reference for more information.
    show_progress: bool = True
        Show progress bar.

    RETURN
    ------
    Dict[str, Any]
    """
    trad_accum = TraditionalCriteriaAccumulator()
    cur_accum = CurrentCriteriaAccumulator(alpha=alpha, beta=beta)
    gt_files = {
        str(v.name).replace("%s.json" % gt_name_suffix, ""): v
        for v in Path(gt_dir).glob("*%s.json" % gt_name_suffix)}
    pred_files = {
        str(v.name).replace("%s.json" % pred_name_suffix, ""): v
        for v in Path(pred_dir).glob("*%s.json" % pred_name_suffix)}
    _omit_gt = set(gt_files.keys()) - set(pred_files.keys())
    for k in _omit_gt:
        del gt_files[k]
        print("'%s' is omitted from groundtruth files" % k)
    _omit_pred = set(pred_files.keys()) - set(gt_files.keys())
    for k in _omit_pred:
        del pred_files[k]
        print("'%s' is omitted from prediction files" % k)
    logging.info(
        "Obtained %d groundtruth and prediction file pairs", len(gt_files))
    if show_progress:
        ks = tqdm(gt_files, desc="Processing")
    else:
        ks = gt_files
    for k in ks:
        logging.info("Processing '%s'", k)
        with open(gt_files[k], "r") as fp:
            gt_annos = data_schema.data_parser(
                json.load(fp), gt_score_maps_root_dir)
        with open(pred_files[k], "r") as fp:
            pred_annos = data_schema.data_parser(
                json.load(fp), pred_score_maps_root_dir)
        if gt_annos.frames_count != pred_annos.frames_count:
            raise ValueError("Frames count Pred != frames count GT")
        trad_accum.update(preds=pred_annos, gts=gt_annos)
        cur_accum.update(preds=pred_annos, gts=gt_annos)
    results = {}
    results.update(trad_accum.summarize())
    results.update(cur_accum.summarize())
    return results
