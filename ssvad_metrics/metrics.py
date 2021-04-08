import json

from ssvad_metrics.criteria import current_criteria, traditional_criteria
from ssvad_metrics import data_schema


def evaluate(
        gt_path: str,
        pred_path: str,
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
    alpha: float = 0.1
        A threshold used in NTPT calculation. See reference for more information.
    beta: float = 0.1
        A threshold used in NTP, NFP, and NTPT calculations. See reference for more information.

    RETURN
    ------
    Dict[str, Any]
    """
    with open(gt_path, "r") as fp:
        gt_annos = data_schema.data_parser(json.load(fp))
    with open(pred_path, "r") as fp:
        pred_annos = data_schema.data_parser(json.load(fp))
    if gt_annos.frames_count != pred_annos.frames_count:
        raise ValueError("Frames count Pred != frames count GT")
    results = {}
    results.update(
        traditional_criteria(pred_annos, gt_annos)
    )
    results.update(
        current_criteria(pred_annos, gt_annos, alpha=alpha, beta=beta))
    return results
