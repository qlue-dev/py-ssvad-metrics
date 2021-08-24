import argparse
import os
from pathlib import Path

import cv2
import pandas as pd
import math
from ssvad_metrics.data_schema import VADAnnotation, VADFrame, AnomalousRegion


def main(args):
    anno_files = list(Path(args.dir).rglob("*.txt"))
    assert len(anno_files) > 0
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
    for anno_fpath in anno_files:
        print(anno_fpath.parent)
        jpg_files = list(Path(anno_fpath.parent).glob("*.jpg"))
        img = cv2.imread(str(jpg_files[0]), cv2.IMREAD_COLOR)
        img_h, img_w = img.shape[:2]
        _d = dict(
            frames_count=len(jpg_files),
            is_anomalous_regions_available=True,
            is_anomaly_track_id_available=True,
            video_length_sec=None,
            frame_width=img_w,
            frame_height=img_h,
            frame_rate=15.0,  # ref: https://ieeexplore.ieee.org/document/9093457/
            frames=[]
        )
        annos_df = pd.read_csv(
            anno_fpath, sep=" ", index_col=False, header=None, names=["filename", "T", "x", "y", "w", "h"])
        annos_df["frame_id"] = annos_df["filename"].apply(
            lambda x: int(os.path.splitext(os.path.split(x)[1])[0]))
        annos_df = annos_df.sort_values("frame_id")
        annos_df_grouped = {
            name: group for name, group in annos_df.groupby("frame_id")}
        for frame_id in range(_d["frames_count"]):
            frame_id += 1  # since frame index start from 1
            try:
                _f = annos_df_grouped[frame_id]
            except KeyError:
                _f = None
            if _f is None:
                frame = VADFrame(
                    frame_id=frame_id,
                    frame_filename="%05d.jpg" % int(frame_id),
                    video_time_sec=None,
                    anomaly_track_id=-1,
                    frame_level_score=None,
                    anomalous_regions=[]
                )
            else:
                anomalous_regions = [
                    AnomalousRegion(
                        bounding_box=[
                            math.floor(row["x"]-row["w"]/2),
                            math.floor(row["y"]-row["h"]/2),
                            math.ceil(row["x"]+row["w"]/2),
                            math.ceil(row["y"]+row["h"]/2)
                        ],
                        score=1.0
                    )
                    for _, row in _f.iterrows()
                ]
                frame = VADFrame(
                    frame_id=frame_id,
                    frame_filename=_f.iloc[0]["filename"],
                    video_time_sec=None,
                    anomaly_track_id=int(_f.iloc[0]["T"]),
                    frame_level_score=None,
                    anomalous_regions=anomalous_regions
                )
            _d["frames"].append(frame)
        out = os.path.join(args.outdir, os.path.splitext(
            anno_fpath.name)[0] + ".json")

        vad_anno = VADAnnotation(**_d)
        with open(out, "w") as fp:
            fp.write(vad_anno.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dir", required=True,
        help="Path to the Test directory containing subdirectory that contains JPEG files and a txt GT")
    parser.add_argument(
        "-o", "--outdir", default="output",
        help="Path to output directory"
    )
    args = parser.parse_args()
    main(args)
