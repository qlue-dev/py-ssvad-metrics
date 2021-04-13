import argparse
import os
from pathlib import Path

import cv2
import pandas as pd
import ssvad_metrics


def main(args):
    anno_files = list(Path(args.annos).glob("*.txt"))
    assert len(anno_files) > 0
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
    for anno_fpath in anno_files:
        nn = os.path.splitext(anno_fpath.name)[0][-5:-3]
        video_path = os.path.join(args.dir, nn + ".avi")
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("can not open")
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        _d = dict(
            frames_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            is_anomalous_regions_available=True,
            is_anomaly_track_id_available=True,
            video_length_sec=cap.get(cv2.CAP_PROP_POS_MSEC),
            frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_rate=cap.get(cv2.CAP_PROP_FPS),
            frames=[]
        )
        print(_d)
        vad_anno = ssvad_metrics.data_schema.VADAnnotation(**_d)
        annos_df = pd.read_csv(
            anno_fpath, sep=" ", index_col=False, header=None, names=["filename", "T", "x", "y", "w", "h"])
        annos_df["frame_id"] = annos_df["filename"].apply(
            lambda x: int(os.path.splitext(x)[0]))
        annos_df = annos_df.sort_values("frame_id")
        annos_df_grouped = {
            name: group for name, group in annos_df.groupby("frame_id")}
        for frame_id in range(vad_anno["frames_count"]):
            frame_id += 1  # since frame index start from 1
            try:
                _f = annos_df_grouped[frame_id]
            except KeyError:
                _f = None
            if _f is None:
                frame = ssvad_metrics.data_schema.VADFrame(
                    frame_id=frame_id,
                    frame_filename=None,
                    video_time_sec=None,
                    anomaly_track_id=-1,
                    frame_level_score=None,
                    anomalous_regions=[]
                )
            else:
                anomalous_regions = [
                    ssvad_metrics.data_schema.data_schema.AnomalousRegion(
                        bounding_box=[
                            int(row["x"]),
                            int(row["y"]),
                            int(row["x"] + row["w"]),
                            int(row["y"] + row["h"])
                        ],
                        score=1.0
                    )
                    for _, row in _f.iterrows()
                ]
                frame = ssvad_metrics.data_schema.VADFrame(
                    frame_id=frame_id,
                    frame_filename=_f.iloc[0]["filename"],
                    video_time_sec=None,
                    anomaly_track_id=int(_f.iloc[0]["T"]),
                    frame_level_score=None,
                    anomalous_regions=anomalous_regions
                )
            vad_anno.frames.append(frame)
        out = os.path.join(args.outdir, os.path.splitext(
            anno_fpath.name)[0] + ".json")

        with open(out, "w") as fp:
            fp.write(vad_anno.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dir", required=True,
        help="Path to the directory containing AVI files")
    parser.add_argument(
        "-a", "--annos", required=True,
        help="Path to the annotations (from street scene) of given Test files"
    )
    parser.add_argument(
        "-o", "--outdir", default="output",
        help="Path to output directory"
    )
    args = parser.parse_args()
    main(args)
