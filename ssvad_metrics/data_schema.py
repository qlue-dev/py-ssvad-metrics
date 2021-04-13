from typing import Dict, List, Optional

import pandas as pd
from numpy.core.arrayprint import FloatingFormat
from pydantic import (BaseModel, PositiveFloat, PositiveInt, ValidationError,
                      confloat, conlist, validator)
from pydantic.fields import Field
from pydantic.types import conint

# def load_motchallenge(fname, **kwargs):
#     """Load MOT challenge data.
#     Borrowed from [py-motmetrics](https://github.com/cheind/py-motmetrics/blob/447c622762476295bdb76965aaa655072e6f9690/motmetrics/io.py#L48)

#     Params
#     ------
#     fname : str
#         Filename to load data from.

#     Kwargs
#     ------
#     sep : str
#         Allowed field separators, defaults to '\s+|\t+|,'
#     min_confidence : float
#         Rows with confidence less than this threshold are removed.
#         Defaults to -1. You should set this to 1 when loading
#         ground truth MOTChallenge data, so that invalid rectangles in
#         the ground truth are not considered during matching.
#     Returns
#     ------
#     df : pandas.DataFrame
#         The returned dataframe has the following columns
#             'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'
#         The dataframe is indexed by ('FrameId', 'Id')
#     """

#     sep = kwargs.pop('sep', r'\s+|\t+|,')
#     min_confidence = kwargs.pop('min_confidence', -1)
#     # df = pd.read_csv(
#     #     fname,
#     #     sep=sep,
#     #     index_col=[0, 1],
#     #     skipinitialspace=True,
#     #     header=None,
#     #     names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height',
#     #            'Confidence', 'ClassId', 'Visibility', 'unused'],
#     #     engine='python'
#     # )
#     df = pd.read_csv(
#         fname,
#         sep=sep,
#         skipinitialspace=True,
#         header=None,
#         names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height',
#                'Confidence', 'ClassId', 'Visibility', 'unused'],
#         engine='python'
#     )

#     # Account for matlab convention.
#     # df[['X', 'Y']] -= (1, 1)

#     # Removed trailing column
#     del df['unused']

#     # Remove all rows without sufficient confidence
#     df = df[df['Confidence'] >= min_confidence]
#     return df

# we borrow MOT format
# https://motchallenge.net/instructions/
# https://github.com/openvinotoolkit/cvat/blob/develop/cvat/apps/dataset_manager/formats/README.md#mot-dumper

# def load_vatracks(fname: str) -> Dict[int, list]:
#     """
#     Load video anomaly tracks data.

#     PARAMETERS
#     ----------
#     fname: str
#         File name to the video anomaly tracks data.
#         The file must not have header, containing:
#         AnomalyTrackId,FrameId

#     RETURNS
#     -------
#     Dict[int, list]
#         A dictionary containing AnomalyTrackId as the key,
#         and list of FrameId as the value.
#     """
#     df = pd.read_csv(
#         fname,
#         skipinitialspace=True,
#         header=None,
#         names=['AnomalyTrackId', 'FrameId'],
#         engine='python'
#     )
#     va_tracks = {}
#     for _, row in df.iterrows():
#         va_tracks.setdefault(
#             int(row["AnomalyTrackId"]), list()
#         ).append(row["FrameId"])
#     return va_tracks


class AnomalousRegion(BaseModel):
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
    frame_filename: Optional[str] = None
    video_time_sec: Optional[PositiveFloat] = None
    anomaly_track_id: Optional[int] = Field(
        ..., description=(
            "Set to None if anomalous track not available. "
            "Set to -1 for Negative."))
    frame_level_score: Optional[confloat(ge=0., le=1.0)] = Field(
        ..., description=(
            "Set to None if anomalous region available. "
            "If None, anomalous_regions must not None. "
            "For GT, 1 for Positive and 0 for Negative."))
    anomalous_regions: List[AnomalousRegion] = Field(
        ..., description=(
            "Set to None if anomalous region not available. "
            "If empty, the frame is considered as Negative."))


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
