import numpy as np
import ssvad_metrics


def test_evaluate_gt_pred():
    result = ssvad_metrics.metrics.evaluate(
        "tests/gt_examples/Test001_gt.json",
        "tests/pred_examples/Test001_pred.json")
    print(result)


def test_accumulated_evaluate_gt_pred():
    result = ssvad_metrics.metrics.accumulated_evaluate(
        "tests/gt_examples",
        "tests/pred_examples",
        gt_name_suffix="_gt",
        pred_name_suffix="_pred")
    print(result)


def test_allclose():
    result = ssvad_metrics.metrics.evaluate(
        "tests/gt_examples/Test001_gt.json",
        "tests/pred_examples/Test001_pred.json")
    result_accum = ssvad_metrics.metrics.accumulated_evaluate(
        "tests/gt_examples",
        "tests/pred_examples",
        gt_name_suffix="_gt",
        pred_name_suffix="_pred")
    assert np.allclose(list(result.values()), list(result_accum.values()))
