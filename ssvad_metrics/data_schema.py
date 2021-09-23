"""
Data structures used by the metrics
"""
import os
from typing import List, Optional

import cv2
import numpy as np
from pydantic import (BaseModel, Field, PositiveFloat,
                      confloat, conint, conlist, validator)

PX_MAP_EXTS = [".tiff", ".npy"]


def load_pixel_score_map(p: str) -> np.ndarray:
    """
    Load and validate pixel score map array.
    """
    ext = os.path.splitext(p)[1]
    if ext == ".tiff":
        arr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    elif ext == ".npy":
        arr = np.load(p)
    else:
        # unsupported ext
        raise ValueError(
            "Unsupported file extension '%s' for anomaly score map file!" % ext)
    if arr.dtype != np.float32:
        raise ValueError((
            "Pixel score map array must have np.float32 (single precision floating point) "
            "data type! (provided: %s)") % arr.dtype)
    if arr.min() < 0.0 or arr.max() > 1.0:
        raise ValueError((
            "Scores must be in the range of 0.0 to 1.0! "
            "(current scores range: [%.2f, %.2f])"
        ) % (arr.min(), arr.max()))
    if len(arr.shape) != 2:
        raise ValueError(
            "Pixel score map must be 2D array, however %dD array given!" % len(arr.shape))
    return arr


class AnomalousRegion(BaseModel):
    """
    Data structure that represents anomalous region as bounding box.
    """
    bounding_box: conlist(int, min_items=4, max_items=4) = Field(
        ..., description=(
            "[x left, y top, x right, y bottom]. "
            "Must be an absolute type value."))
    score: confloat(ge=0., le=1.0) = Field(
        ..., description=(
            "Score for the region. "
            "For GT, value must be 1.")
    )


class VADFrame(BaseModel):
    frame_id: conint(gt=0)
    frame_filename: Optional[str] = Field(
        None, description="LEGACY. Filename of the frame.")
    frame_filepath: Optional[str] = Field(
        None, description=(
            "Full path (either relative/absolute) "
            "to the image file of the frame.")
    )
    video_time_sec: Optional[PositiveFloat] = None
    anomaly_track_id: Optional[int] = Field(
        None, description=(
            "Set to None if anomalous track not available. "
            "Set to -1 for Negative."))
    frame_level_score: Optional[confloat(ge=0., le=1.0)] = Field(
        None, description=(
            "Frame-level anomaly score. "
            "Ignored if `~VADAnnotation.is_anomalous_regions_available` is True. "
            "Set to None if not available."))
    anomalous_regions: Optional[List[AnomalousRegion]] = Field(
        None, description=(
            "Bounding boxes representing anomalous regions. "
            "If a frame does not contain any bboxes, provide empty list. "
            "Ignored if pixel_level_scores_map is available. "
            "Set to None if not available."))
    pixel_level_scores_map: Optional[str] = Field(
        None, description=(
            "Path to the pixel anomaly scores map file. "
            "Scores must be in the range of 0.0 to 1.0 "
            "with data type np.float32 (single-precision floating point). "
            "Supported file formats: %s. "
            "For ground-truth, it can be np.bool instead (boolean), but only npy files supported. "
            "Set to None if not available.") % PX_MAP_EXTS)

    @validator('pixel_level_scores_map')
    def pixel_level_scores_map_file_exist(cls, v, **kwargs):
        if v is None:
            return v
        v = os.path.expandvars(os.path.expanduser(v))
        ext = os.path.splitext(v)[1]
        if not os.path.exists(v):
            raise FileNotFoundError(
                "Anomaly score map file '%s' is not exist!" % v)
        if ext not in PX_MAP_EXTS:
            raise ValueError(
                "Unsupported file extension '%s' for anomaly score map file!" % ext)
        return v


class VADAnnotation(BaseModel):
    is_gt: Optional[bool] = Field(
        None, description="Assuring the chosen file is groundtruth file or prediction file.")
    frames_count: conint(gt=0)
    is_anomalous_regions_available: bool = Field(
        ..., description="Set to true if each frame contain pixel_level_scores_map or anomalous_regions.")
    is_anomaly_track_id_available: bool = Field(
        ..., description="Set to true if track_id is available.")
    video_length_sec: Optional[PositiveFloat] = None
    frame_width: conint(gt=0)
    frame_height: conint(gt=0)
    frame_rate: Optional[PositiveFloat] = None
    frames: List[VADFrame] = Field(
        ..., description="Length of 'frames' must match 'frames_count'")

    @validator('frames')
    def frames_len(cls, v, values, **kwargs):
        if len(v) != values["frames_count"]:
            raise ValueError(
                "Length of 'frames' does not match 'frames_count'")
        return v

    @validator('frames', each_item=True)
    def anno_available(cls, v, **kwargs):
        if v.pixel_level_scores_map is None and v.anomalous_regions is None and v.frame_level_score is None:
            raise ValueError(
                "Neither frame_level_score nor anomalous_regions nor pixel_level_scores_map is provided!")
        return v

    class Config:
        extra = "allow"


def data_parser(annos: dict, score_maps_root_dir: Optional[str]) -> VADAnnotation:
    """
    Parse data.
    Also, join `score_maps_root_dir` to `VADFrame.pixel_level_scores_map`.
    """
    if score_maps_root_dir is not None:
        for f in annos["frames"]:
            if not isinstance(f["pixel_level_scores_map"], str):
                continue
            f["pixel_level_scores_map"] = os.path.join(
                score_maps_root_dir, f["pixel_level_scores_map"])
    return VADAnnotation(**annos)
