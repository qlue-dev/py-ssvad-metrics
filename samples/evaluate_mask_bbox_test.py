import numpy as np
import ssvad_metrics


def test_evaluate_gt_mask_pred_single():
    result = ssvad_metrics.metrics.evaluate(
        "samples/samples_mask_pred_bbox_gt/Test001_gt.json",
        "samples/samples_mask_pred_bbox_gt/Test001_pred.json")
    print(result)
