import json
import logging
import os
from functools import partial
from pathlib import Path
from typing import Optional

import cv2
import ffmpeg
import numpy as np
from matplotlib import cm
from tqdm import tqdm

from ssvad_metrics.data_schema import (VADAnnotation, VADFrame, data_parser,
                                       load_pixel_score_map)
from ssvad_metrics.utils import connected_components, image_compositing

VIDEO_EXT = ".mp4"
logger = logging.getLogger()


def _open_image(vad_frame: VADFrame, images_root_dir: Optional[str] = None) -> np.ndarray:
    p = vad_frame.frame_filepath
    if not p:
        raise ValueError(
            "The given ground-truth frame has no frame_filepath! To ignore, set show_image to False.")
    if images_root_dir is not None:
        p = os.path.join(images_root_dir, p)
    if not os.path.isfile(p):
        raise FileNotFoundError("Can not open image file '%s'!" % p)
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if len(img.shape) > 3:
        img = img[:, :, :3]
    return img


def _draw_frame(
        show_image: bool,
        line_thickness: int,
        text_scale: float,
        text_thickness: float,
        gts: VADAnnotation,
        pred_frm_shp: tuple,
        gt_frm_shp: tuple,
        img: Optional[np.ndarray],
        pred_frm: VADFrame,
        gt_frm: VADFrame,
        overlay_alpha: float,
        threshold: float) -> np.ndarray:
    if img is None:
        img = np.ones(gt_frm_shp + (3,), dtype=np.uint8) * 255
    # Draw Pred Mask
    pred_frm_scr = None
    if pred_frm.pixel_level_scores_map is not None:
        pred_m = load_pixel_score_map(pred_frm.pixel_level_scores_map)
        if pred_m.shape != pred_frm_shp:
            raise ValueError((
                "The loaded predictions anomaly score map frame shape %s "
                "mismatched with the frame shape "
                "defined in the annotation %s!") % (pred_m.shape, pred_frm_shp))
        pred_cm = (cm.RdYlBu(pred_m)[:, :, :3] * 255).astype(np.uint8)
        if show_image:
            img_a = np.ones(img.shape[:2], dtype=np.float32)
            pred_m_a = img_a * overlay_alpha
            pred_m_a[pred_m < threshold] = 0.
            img = image_compositing(pred_cm, img, pred_m_a, img_a)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            pred_cm[pred_m < threshold] = 255
            img = pred_cm
    elif pred_frm.anomalous_regions is not None:
        for ar in pred_frm.anomalous_regions:
            if ar.score < threshold:
                continue
            bb = ar.bounding_box
            cv2.rectangle(
                img, bb[:2], bb[2:], color=(0, 0, 255),
                thickness=line_thickness)
            text_org = np.asarray(bb[:2]) + np.asarray([0, bb[3]])
            cv2.putText(
                img, str(ar.score), text_org,
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, color=(
                    0, 0, 255),
                thickness=text_thickness)
    elif pred_frm.frame_level_score is not None:
        pred_frm_scr = pred_frm.frame_level_score
    else:
        raise ValueError((
            "The predicition file has no 'pixel_level_scores_map' "
            "nor 'anomalous_regions' nor 'frame_level_score' "
            "available!"))
    # Draw GT
    gt_frm_scr = None
    if gt_frm.pixel_level_scores_map is not None:
        gt_m = load_pixel_score_map(gt_frm.pixel_level_scores_map)
        if gt_m.shape != gt_frm_shp:
            raise ValueError((
                "The loaded ground-truth anomaly score map frame shape %s "
                "mismatched with the frame shape "
                "defined in the annotation %s!") % (gt_m.shape, gt_frm_shp))
        gt_m = gt_m.astype(np.bool) * np.uint8(255)
        gt_num_ccs, gt_cc_labels = connected_components(gt_m)
        for k in range(1, gt_num_ccs):  # skip BG
            gt_ar_m = gt_cc_labels == k
            cntr = cv2.findContours(
                gt_ar_m.astype(np.uint8), mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_SIMPLE)[0][0]
            cv2.polylines(
                img, cntr, isClosed=True,
                color=(0, 255, 0), thickness=line_thickness)
    elif gt_frm.anomalous_regions is not None:
        for ar in gt_frm.anomalous_regions:
            bb = ar.bounding_box
            cv2.rectangle(
                img, bb[:2], bb[2:], color=(0, 255, 0),
                thickness=line_thickness)
    elif gt_frm.frame_level_score is not None:
        gt_frm_scr = bool(gt_frm.frame_level_score)
    else:
        raise ValueError((
            "The ground-truth file has no 'pixel_level_scores_map' "
            "nor 'anomalous_regions' nor 'frame_level_score' "
            "available!"))
    # Draw frame level
    if not (pred_frm_scr is None and gt_frm_scr is None):
        texts = []
        if pred_frm_scr is not None:
            texts.append("P:%.3f" % pred_frm_scr)
        if gt_frm_scr is not None:
            texts.append("GT:POS" if gt_frm_scr else "GT:NEG")
        texts = " | ".join(texts)
        cv2.putText(
            img, texts, (5, gts.frame_height),
            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 127, 255),
            thickness=text_thickness)
    # Dump video
    return img


def visualize(
        gt_path: str,
        pred_path: str,
        out_dir: Optional[str] = None,
        video_fps: Optional[float] = None,
        gt_score_maps_root_dir: Optional[str] = None,
        pred_score_maps_root_dir: Optional[str] = None,
        images_root_dir: Optional[str] = None,
        show_image: bool = True,
        line_thickness: int = 1,
        text_scale: float = 1.0,
        text_thickness: float = 1.0,
        overlay_alpha: float = 0.4,
        threshold: float = 0.,
        **kwargs) -> str:
    """
    Visualize the single-scene video anomaly detection
    predictions and ground-truths.

    Prediction score maps are drawn as an overlay masks with opacity `overlay_alpha`
    using RdYlBu colormap (red for 1.0, blue for 0.0). If predictions are using bounding boxes
    instead, it will be always drawn as red colored bbox with its score. If both are unavailable
    (frame-level only), then the frame-level score will be drawn
    on the bottom-left of the image (P:score).

    Ground-truth anomalous maps are drawn as contour lines with green colors.
    If ground-truth are using bounding boxes instead,
    it will be always drawn as green colored bbox.
    If both are unavailable (frame-level only), then the frame-level ground-truth will be drawn
    on the bottom-left of the image (GT:NEG or GT:POS).

    Any score below `threshold` in the score map will totally transparent.
    Overlay masks will always be drawn as solid colors (no transparency) when `show_image` is `False`.
    Solid white color will be used as a background if `show_image` is `False` and no overlay masks.

    The visualization result will be saved as a video file.
    NOTE: requires FFMPEG installation.

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
    out_dir: Optional[str] = None
        Directory for the video file output.
        If None, then it is the same as `pred_path` except the extension (mp4).
    video_fps: Optional[float] = None
        Force frame-rate of the video.
        By default, it will use frame-rate defined in the ground-truth JSON file.
        If frame-rate is not defined in the ground-truth JSON file, then 25 fps will be used.
    gt_score_maps_root_dir: Optional[str] = None
        The root directory for the pixel-level anomaly scores maps files in the ground-truth JSON.
    pred_score_maps_root_dir: Optional[str] = None
        The root directory for the pixel-level anomaly scores maps files in the prediction JSON.
    images_root_dir: Optional[str] = None
        The root directory for the frame files in the ground-truth JSON.
    show_image: bool = True
        Show background image. Frames in the ground-truth file must contain `frame_filepath`.
    line_thickness: int = 1
        Thickness of the line.
    text_scale: float = 1.0
        Scale of the text.
    text_thickness: float = 1.0
        Thickness of the text.
    overlay_alpha: float = 0.4
        Overlay opacity (1 - transparency).
    threshold: float = 0.
        Any score below `threshold` in the score map will
        totally transparent. When `show_image` is `False`,
        they will be set to solid white instead.
        If it is bounding boxes, omit bounding boxes that
        have score below `threshold`.

    RETURN
    ------
    str
        Path of the output video.
    """
    if overlay_alpha < 0. or overlay_alpha > 1.:
        raise ValueError("overlay_alpha must be in range [0 .. 1]!")
    with open(gt_path, "r") as fp:
        gts = data_parser(
            json.load(fp), gt_score_maps_root_dir)
    with open(pred_path, "r") as fp:
        preds = data_parser(
            json.load(fp), pred_score_maps_root_dir)
    if gts.frames_count != preds.frames_count:
        raise ValueError("Frames count Pred != frames count GT")
    if preds.is_gt:
        raise ValueError(
            "The given prediction file has is_gt=True flag!")
    if gts.is_gt is not None and not gts.is_gt:
        raise ValueError(
            "The given ground-truth file has is_gt=False flag!")
    pred_frm_shp = (preds.frame_height, preds.frame_width)
    gt_frm_shp = (gts.frame_height, gts.frame_width)
    if pred_frm_shp != gt_frm_shp:
        raise ValueError((
            "Predictions frame shape %s mismatched "
            "with the ground-truth %s!" % (pred_frm_shp, gt_frm_shp)))
    # Obtain background images iterator
    if show_image:
        images = map(
            partial(_open_image, images_root_dir=images_root_dir),
            gts.frames)
    else:
        images = [None for _ in range(len(gts.frames))]
    # Video sink
    if out_dir is None:
        vsnk_p = os.path.splitext(pred_path)[0] + VIDEO_EXT
    else:
        vsnk_p = os.path.join(
            out_dir,
            os.path.splitext(os.path.split(pred_path)[1])[0] + VIDEO_EXT)
    if video_fps is None:
        video_fps = gts.frame_rate
    if video_fps is None or video_fps <= 0.:
        video_fps = 25.
    # vsnk = cv2.VideoWriter(
    #     vsnk_p,
    #     fourcc=cv2.VideoWriter_fourcc(*"MJPG"),
    #     fps=video_fps,
    #     frameSize=(gts.frame_width, gts.frame_height)
    # )
    # if not vsnk.isOpened():
    #     raise IOError("Unable to create VideoWriter on '%s'!" % vsnk_p)

    vsnk = (
        ffmpeg
        .input(
            'pipe:',
            hide_banner=None, loglevel="fatal",
            format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(gts.frame_width, gts.frame_height))
        .output(vsnk_p, crf=18, preset='faster', movflags='faststart', vcodec="h264", pix_fmt='yuv420p')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    # Draw frame by frame
    for img, pred_frm, gt_frm in zip(images, preds.frames, gts.frames):
        img = _draw_frame(
            show_image, line_thickness, text_scale, text_thickness,
            gts, pred_frm_shp, gt_frm_shp, img, pred_frm, gt_frm, overlay_alpha, threshold)
        # vsnk.write(img)
        vsnk.stdin.write(
            img
            .astype(np.uint8)
            .tobytes()
        )
    # release video
    vsnk.stdin.close()
    vsnk.wait()
    return vsnk_p


def _visualize_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="""Visualize the single-scene video anomaly detection
predictions and ground-truths.

Prediction score maps are drawn as an overlay masks with opacity `overlay_alpha`
using RdYlBu colormap (red for 1.0, blue for 0.0). If predictions are using bounding boxes
instead, it will be always drawn as red colored bbox with its score. If both are unavailable
(frame-level only), then the frame-level score will be drawn
on the bottom-left of the image (P:score).

Ground-truth anomalous maps are drawn as contour lines with green colors.
If ground-truth are using bounding boxes instead,
it will be always drawn as green colored bbox.
If both are unavailable (frame-level only), then the frame-level ground-truth will be drawn
on the bottom-left of the image (GT:NEG or GT:POS).

Any score below `threshold` in the score map will totally transparent.
Overlay masks will always be drawn as solid colors (no transparency) when `show_image` is `False`.
Solid white color will be used as a background if `show_image` is `False` and no overlay masks.

The visualization result will be saved as a video file.
NOTE: requires FFMPEG installation.""",
        allow_abbrev=True
    )
    parser.add_argument(
        "gt_path",
        type=str,
        help="""Path to VADAnnotation-formatted JSON file containing the ground truth annotation
of the video anomaly detection. See 'data_schema.VADAnnotation.schema()' or
'data_schema.VADAnnotation.schema_json()' for the JSON schema."""
    )
    parser.add_argument(
        "pred_path",
        type=str,
        help="""Path to VADAnnotation-formatted JSON file containing the prediction results
of the video anomaly detection. See 'data_schema.VADAnnotation.schema()' or
'data_schema.VADAnnotation.schema_json()' for the JSON schema."""
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=False,
        default=None,
        help="""Directory for the video file output.
If None, then it is the same as 'pred_path' except the extension (mp4)."""
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        required=False,
        default=None,
        help="""Force frame-rate of the video.
By default, it will use frame-rate defined in the ground-truth JSON file.
If frame-rate is not defined in the ground-truth JSON file, then 25 fps will be used."""
    )
    parser.add_argument(
        "--gt_score_maps_root_dir",
        type=str,
        required=False,
        default=None,
        help="The root directory for the pixel-level anomaly scores maps files in the ground-truth JSON."
    )
    parser.add_argument(
        "--pred_score_maps_root_dir",
        type=str,
        required=False,
        default=None,
        help="The root directory for the pixel-level anomaly scores maps files in the prediction JSON."
    )
    parser.add_argument(
        "--images_root_dir",
        type=str,
        required=False,
        default=None,
        help="The root directory for the frame files in the ground-truth JSON."
    )
    parser.add_argument(
        "--line_thickness",
        type=int,
        required=False,
        default=1,
        help="Thickness of the line."
    )
    parser.add_argument(
        "--text_scale",
        type=float,
        required=False,
        default=1.0,
        help="Scale of the text."
    )
    parser.add_argument(
        "--text_thickness",
        type=float,
        required=False,
        default=1.0,
        help="Thickness of the text."
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        required=False,
        default=0.4,
        help="Overlay opacity (1 - transparency)."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        default=0.,
        help="""Any score below 'threshold' in the score map will
totally transparent. When 'show_image' is 'False',
they will be set to solid white instead.
If it is bounding boxes, omit bounding boxes that
have score below 'threshold'."""
    )
    parser.add_argument(
        "--show_image",
        action="store_true",
        help="Show background image. Frames in the ground-truth file must contain 'frame_filepath'."
    )
    args = parser.parse_args()
    p = visualize(**vars(args))
    print("Video file saved into `%s`." % p)


def visualize_dir(
        gt_dir: str,
        pred_dir: str,
        out_dir: Optional[str] = None,
        video_fps: Optional[float] = None,
        gt_name_suffix: str = "",
        pred_name_suffix: str = "",
        gt_score_maps_root_dir: Optional[str] = None,
        pred_score_maps_root_dir: Optional[str] = None,
        images_root_dir: Optional[str] = None,
        show_image: bool = True,
        line_thickness: int = 1,
        text_scale: float = 1.0,
        text_thickness: float = 1.0,
        overlay_alpha: float = 0.4,
        threshold: float = 0.,
        show_progress: bool = True,
        **kwargs) -> None:
    """
    Visualize the single-scene video anomaly detection
    predictions and ground-truths.

    Prediction score maps are drawn as an overlay masks with opacity `overlay_alpha`
    using RdYlBu colormap (red for 1.0, blue for 0.0). If predictions are using bounding boxes
    instead, it will be always drawn as red colored bbox with its score. If both are unavailable
    (frame-level only), then the frame-level score will be drawn
    on the bottom-left of the image (P:score).

    Ground-truth anomalous maps are drawn as contour lines with green colors.
    If ground-truth are using bounding boxes instead,
    it will be always drawn as green colored bbox.
    If both are unavailable (frame-level only), then the frame-level ground-truth will be drawn
    on the bottom-left of the image (GT:NEG or GT:POS).

    Any score below `threshold` in the score map will totally transparent.
    Overlay masks will always be drawn as solid colors (no transparency) when `show_image` is `False`.
    Solid white color will be used as a background if `show_image` is `False` and no overlay masks.

    The visualization result will be saved as a video file.
    NOTE: requires FFMPEG installation.

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
    out_dir: Optional[str] = None
        Directory for the video file output.
        If None, then it is the same as `pred_path` except the extension (mp4).
    video_fps: Optional[float] = None
        Force frame-rate of the video.
        By default, it will use frame-rate defined in the ground-truth JSON file.
        If frame-rate is not defined in the ground-truth JSON file, then 25 fps will be used.
    gt_name_suffix: str = ""
        Fixed file name suffix, if any. Do not include the file extension.
    pred_name_suffix: str = ""
        Fixed file name suffix, if any. Do not include the file extension.
    gt_score_maps_root_dir: Optional[str] = None
        The root directory for the pixel-level anomaly scores maps files in the ground-truth JSON.
    pred_score_maps_root_dir: Optional[str] = None
        The root directory for the pixel-level anomaly scores maps files in the prediction JSON.
    images_root_dir: Optional[str] = None
        The root directory for the frame files in the ground-truth JSON.
    show_image: bool = True
        Show background image. Frames in the ground-truth file must contain `frame_filepath`.
    line_thickness: float = 1.0
        Thickness of the line.
    text_scale: float = 1.0
        Scale of the text.
    text_thickness: float = 1.0
        Thickness of the text.
    overlay_alpha: float = 0.4
        Overlay opacity (1 - transparency).
    threshold: float = 0.
        Any score below `threshold` in the score map will
        totally transparent. When `show_image` is `False`,
        they will be set to solid white instead.
        If it is bounding boxes, omit bounding boxes that
        have score below `threshold`.
    show_progress: bool = True
        Show progress bar.
    """
    gt_files = {
        str(v.name).replace("%s.json" % gt_name_suffix, ""): str(v)
        for v in Path(gt_dir).glob("*%s.json" % gt_name_suffix)}
    pred_files = {
        str(v.name).replace("%s.json" % pred_name_suffix, ""): str(v)
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
        visualize(
            gt_files[k],
            pred_files[k],
            out_dir=out_dir,
            video_fps=video_fps,
            gt_score_maps_root_dir=gt_score_maps_root_dir,
            pred_score_maps_root_dir=pred_score_maps_root_dir,
            images_root_dir=images_root_dir,
            show_image=show_image,
            line_thickness=line_thickness,
            text_scale=text_scale,
            text_thickness=text_thickness,
            overlay_alpha=overlay_alpha,
            threshold=threshold
        )


def _visualize_dir_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="""Visualize the single-scene video anomaly detection
predictions and ground-truths.

Prediction score maps are drawn as an overlay masks with opacity `overlay_alpha`
using RdYlBu colormap (red for 1.0, blue for 0.0). If predictions are using bounding boxes
instead, it will be always drawn as red colored bbox with its score. If both are unavailable
(frame-level only), then the frame-level score will be drawn
on the bottom-left of the image (P:score).

Ground-truth anomalous maps are drawn as contour lines with green colors.
If ground-truth are using bounding boxes instead,
it will be always drawn as green colored bbox.
If both are unavailable (frame-level only), then the frame-level ground-truth will be drawn
on the bottom-left of the image (GT:NEG or GT:POS).

Any score below `threshold` in the score map will totally transparent.
Overlay masks will always be drawn as solid colors (no transparency) when `show_image` is `False`.
Solid white color will be used as a background if `show_image` is `False` and no overlay masks.

The visualization result will be saved as a video file.
NOTE: requires FFMPEG installation.""",
        allow_abbrev=True
    )
    parser.add_argument(
        "gt_dir",
        type=str,
        help="""Path to directory containing VADAnnotation-formatted JSON files.
Each JSON file containing the ground truth annotation
of the video anomaly detection. See 'data_schema.VADAnnotation.schema()' or
'data_schema.VADAnnotation.schema_json()' for the JSON schema."""
    )
    parser.add_argument(
        "pred_dir",
        type=str,
        help="""Path to directory containing VADAnnotation-formatted JSON files.
Each JSON file containing the prediction results
of the video anomaly detection. See 'data_schema.VADAnnotation.schema()' or
'data_schema.VADAnnotation.schema_json()' for the JSON schema."""
    )
    parser.add_argument(
        "--gt_name_suffix",
        type=str,
        required=False,
        default="",
        help="Fixed file name suffix, if any. Do not include the file extension."
    )
    parser.add_argument(
        "--pred_name_suffix",
        type=str,
        required=False,
        default="",
        help="Fixed file name suffix, if any. Do not include the file extension."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=False,
        default=None,
        help="""Directory for the video file output.
If None, then it is the same as 'pred_path' except the extension (mp4)."""
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        required=False,
        default=None,
        help="""Force frame-rate of the video.
By default, it will use frame-rate defined in the ground-truth JSON file.
If frame-rate is not defined in the ground-truth JSON file, then 25 fps will be used."""
    )
    parser.add_argument(
        "--gt_score_maps_root_dir",
        type=str,
        required=False,
        default=None,
        help="The root directory for the pixel-level anomaly scores maps files in the ground-truth JSON."
    )
    parser.add_argument(
        "--pred_score_maps_root_dir",
        type=str,
        required=False,
        default=None,
        help="The root directory for the pixel-level anomaly scores maps files in the prediction JSON."
    )
    parser.add_argument(
        "--images_root_dir",
        type=str,
        required=False,
        default=None,
        help="The root directory for the frame files in the ground-truth JSON."
    )
    parser.add_argument(
        "--line_thickness",
        type=int,
        required=False,
        default=1,
        help="Thickness of the line."
    )
    parser.add_argument(
        "--text_scale",
        type=float,
        required=False,
        default=1.0,
        help="Scale of the text."
    )
    parser.add_argument(
        "--text_thickness",
        type=float,
        required=False,
        default=1.0,
        help="Thickness of the text."
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        required=False,
        default=0.4,
        help="Overlay opacity (1 - transparency)."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        default=0.,
        help="""Any score below 'threshold' in the score map will
totally transparent. When 'show_image' is 'False',
they will be set to solid white instead.
If it is bounding boxes, omit bounding boxes that
have score below 'threshold'."""
    )
    parser.add_argument(
        "--show_image",
        action="store_true",
        help="Show background image. Frames in the ground-truth file must contain 'frame_filepath'."
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="Show progress bar."
    )
    args = parser.parse_args()
    visualize_dir(**vars(args))
    print("Video files saved into `%s`." % args.out_dir)
