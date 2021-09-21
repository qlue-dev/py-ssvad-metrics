import numpy as np
from numpy.__config__ import show

from ssvad_metrics import visualize


def test_visualize_bbox_gt_mask_pred_single_with_image():
    result = visualize(
        "samples/samples_mask_pred_bbox_gt/Test001_gt.json",
        "samples/samples_mask_pred_bbox_gt/Test001_pred.json",
        pred_score_maps_root_dir="samples/samples_mask_pred_bbox_gt",
        images_root_dir="samples/samples_mask_pred_bbox_gt",
        show_image=True)
    print(result)


def test_visualize_bbox_gt_mask_pred_single_no_image():
    result = visualize(
        "samples/samples_mask_pred_bbox_gt/Test001_gt.json",
        "samples/samples_mask_pred_bbox_gt/Test001_pred.json",
        pred_score_maps_root_dir="samples/samples_mask_pred_bbox_gt",
        show_image=False)
    print(result)
