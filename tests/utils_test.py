from ssvad_metrics.data_schema import AnomalousRegion
from ssvad_metrics.utils import bb_s_to_fp32_mask, anomalous_regions_to_float_mask
import pytest
import numpy as np


@pytest.fixture
def dummy_bboxes_scores():
    return (
        np.array([[4, 4, 8, 8], [2, 2, 6, 6]], dtype=np.int32),
        np.array([0.5, 0.3], dtype=np.float32)
    )


def test_bb_s_to_fp32_mask(dummy_bboxes_scores):
    _ = bb_s_to_fp32_mask(
        dummy_bboxes_scores[0],
        dummy_bboxes_scores[1],
        (10, 10)
    )
    print(_)


def test_anomalous_regions_to_float_mask(dummy_bboxes_scores):
    _ = anomalous_regions_to_float_mask([
        AnomalousRegion(
            bounding_box=bb.tolist(),
            score=scr
        ) for bb, scr in zip(*dummy_bboxes_scores)],
        (10, 10))
    print(_)
