import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pydantic import (BaseModel, Field, PositiveFloat, PrivateAttr,
                      ValidationError, confloat, conint, validator)

ANOMALOUS_REGION_EXTS = [".tiff", ".npy"]


def load_anomalous_region(p: str) -> np.ndarray:
    """
    Load and validate anomalous region (pixel score map) array.
    """
    ext = os.path.splitext(p)[1]
    if ext == ".tiff":
        arr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    elif ext == ".npy":
        arr = np.load(p)
    else:
        # unsupported ext
        raise ValueError(
            "Unsupported file extension '%s' for anomalous region file!" % ext)
    assert arr.dtype == np.float32, (
        "Pixel score map array must have np.float32 (single precision floating point) "
        "data type! (provided: %s)") % arr.dtype
    assert arr.min() >= 0.0 and arr.max() <= 1.0, (
        "Scores must be in the range of 0.0 to 1.0! "
        "(current scores range: [%.2f, %.2f])"
    ) % (arr.min(), arr.max())
    return arr


class VADFrame(BaseModel):
    frame_id: conint(gt=0)
    frame_filename: Optional[str] = None
    video_time_sec: Optional[PositiveFloat] = None
    anomaly_track_id: Optional[int] = Field(
        ..., description=(
            "Set to None if anomalous track not available. "
            "Set to -1 for Negative."))
    frame_level_score: Optional[confloat(ge=0., le=1.0)] = Field(
        ..., description=(
            "Set to None if anomalous region is available. "
            "If frame_level_score is None, anomalous_regions must not None. "
            "For GT, 1 for Positive and 0 for Negative."))
    anomalous_region: Optional[str] = Field(
        ..., description=(
            "Path to the anomalous region (pixel score map) file. "
            "Scores must be in the range of 0.0 to 1.0 "
            "with data type np.float32 (single precision floating point). "
            "Supported file formats: %s. "
            "Set to None if anomalous region not available.") % ANOMALOUS_REGION_EXTS)

    @validator('anomalous_region')
    def anomalous_region_file_exist(cls, v, *args, **kwargs):
        v = os.path.expandvars(os.path.expanduser(v))
        ext = os.path.splitext(v)[1]
        if not os.path.exists(v):
            raise ValidationError(
                "Anomalous region file '%s' is not exist!" % v)
        if ext not in ANOMALOUS_REGION_EXTS:
            raise ValueError(
                "Unsupported file extension '%s' for anomalous region file!" % ext)
        return v


class VADAnnotation(BaseModel):
    frames_count: conint(gt=0)
    is_anomalous_regions_available: bool
    is_anomaly_track_id_available: bool
    video_length_sec: Optional[PositiveFloat] = None
    frame_width: conint(gt=0)
    frame_height: conint(gt=0)
    frame_rate: Optional[PositiveFloat] = None
    frames: List[VADFrame] = Field(
        ..., description=("len(frames) == frames_count"))

    @validator('frames')
    def frames_len(cls, v, values, **kwargs):
        if len(v) != values["frames_count"]:
            raise ValidationError(
                "Length of 'frames' does not match 'frames_count'")
        return v

    class Config:
        extra = "allow"


def data_parser(annos: dict) -> VADAnnotation:
    """
    Parse data.
    """
    return VADAnnotation(**annos)
