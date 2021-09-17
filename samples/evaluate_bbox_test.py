import numpy as np
import ssvad_metrics


def test_evaluate_gt_pred_single():
    result = ssvad_metrics.evaluate(
        "samples/samples_bbox_gt/Test001_gt.json",
        "samples/samples_bbox_pred/Test001_pred.json")
    print(result)


def test_accumulated_evaluate_gt_pred():
    result = ssvad_metrics.accumulated_evaluate(
        "samples/samples_bbox_gt",
        "samples/samples_bbox_pred",
        gt_name_suffix="_gt",
        pred_name_suffix="_pred")
    print(result)


def test_allclose():
    result = ssvad_metrics.evaluate(
        "samples/samples_bbox_gt/Test001_gt.json",
        "samples/samples_bbox_pred/Test001_pred.json")
    result_accum = ssvad_metrics.accumulated_evaluate(
        "samples/samples_bbox_gt",
        "samples/samples_bbox_pred",
        gt_name_suffix="_gt",
        pred_name_suffix="_pred")
    assert np.allclose(list(result.values()), list(result_accum.values()))


def test_evaluate_gt_gt_single():
    result = ssvad_metrics.evaluate(
        "samples/samples_bbox_gt/Test001_gt.json",
        "samples/samples_bbox_gt/Test001_gt.json")
    print(result)


def test_evaluate_gt_pnp_single():
    result = ssvad_metrics.evaluate(
        "samples/samples_bbox_gt/Test001_gt.json",
        "samples/samples_bbox_pred/Test001_pred_near_perfect.json")
    print(result)
