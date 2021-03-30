import json

from ssvad_metrics.criteria import current_criteria
from ssvad_metrics.data_schema import data_parser


def compute(gt_path: str, pred_path: str) -> dict:
    """
    Compute performance.
    """
    with open(gt_path, "r") as fp:
        gt_annos = data_parser(json.load(fp))
    with open(pred_path, "r") as fp:
        pred_annos = data_parser(json.load(fp))
    if gt_annos.frames_count != pred_annos.frames_count:
        raise ValueError("Frames count Pred != frames count GT")
    current_criteria(pred_annos, gt_annos)
