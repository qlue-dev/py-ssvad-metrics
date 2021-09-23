# Single-Scene Video Anomaly Detection (SSVAD) Metrics

This project contains evaluation protocol (metrics) for benchmarking single-scene video anomaly detection (SSVAD).

## Evaluation Protocol

This is an unofficial implementation of Sec. 2.2 of [A Survey of Single-Scene Video Anomaly Detection](https://arxiv.org/pdf/2004.05993.pdf).
This metric is intended **only** for single-scene video anomaly detection methods and is untested for other purposes.

There are 3 kind of supported outputs from SSVAD methods/ground-truths:

- Pixel-level anomaly scores maps.
- Bounding-boxes based outputs, it will be implicitly converted into pixel-level anomaly scores maps. Overlapping bounding-boxes will be averaged by its scores.
- Frame-level anomaly scores.

The metrics can be categorized into 4:

1. Track-based metric: requires pixel-level anomaly scores maps predictions and ground-truths, plus anomaly track ID for each frame in the ground-truths that contains anomalous regions (predictions does not require anomaly track ID and is ignored in the process).
1. Region-based metric: requires pixel-level anomaly scores maps predictions and ground-truths.
1. Pixel-level traditional metric: requires pixel-level anomaly scores maps predictions and ground-truths.
1. Frame-level traditional metric: only require frame-level anomaly scores; does not require pixel-level anomaly scores maps predictions and ground-truths.

Each prediction output and ground-truth annotation must be a JSON file that follows data structure defined in [`ssvad_metrics.data_schema.VADAnnotation`](ssvad_metrics/data_schema.py).
Pixel-level anomaly scores maps arrays must be provided using `.tiff` or `.npy` format,
containing single-precision (32-bit) floating point values ranging from 0.0 to 1.0. For ground-truths, it can be same as pixel-level predictions, or using boolean values instead. But only `.npy` files supported if using boolean values (as `.tiff` requires 32-bit floating point values).

## Installation

This metrics is available via PyPI.

```bash
pip install py-ssvad-metrics
```

## Usage

1. Prepare ground-truth JSON files and prediction JSON files (also the pixel anomaly score map linked files for pixel-level predictions and groundtruths).
    - [`ssvad_metrics.data_schema.VADAnnotation`](ssvad_metrics/data_schema.py) can be used for the data structure reference and validator of the JSON file.
    - JSON file examples are in the [samples folder](samples).
    - For UCSD Pedestrian 1 and 2 datasets, CUHK Avenue dataset, and Street Scene dataset, we provided scripts for converting ground-truth annotation files from Street Scene dataset provided by the paper (txt files, each row contains: \<filename> \<track_id> \<x_center> \<y_center> \<width> \<height>). Download link is provided in the paper <http://www.merl.com/demos/video-anomaly-detection>.
1. Example usage for single groundtruth and prediction file pair:

    ```python
    import ssvad_metrics
    result = ssvad_metrics.evaluate(
        "tests/gt_examples/Test001_gt.json",
        "tests/pred_examples/Test001_pred.json")
    ```

1. Example usage for multiple groundtruth and prediction file pairs:

    ```python
    import ssvad_metrics
    result = ssvad_metrics.accumulated_evaluate(
        "tests/gt_examples",
        "tests/pred_examples",
        gt_name_suffix="_gt",
        pred_name_suffix="_pred")
    ```

1. For more examplles, see [samples folder](samples).

## Visual Inspection

We also provide tools for visual inspection for checking the *quality of false positives*.
After installing `py-ssvad-metrics`, the visualizer can be used by executing `ssvad-visualize` or `ssvad-visualize-dir`.
See `ssvad-visualize --help` or `ssvad-visualize-dir --help` for details and usage.
Also, see [`ssvad_metrics.visualize`](ssvad_metrics/visualizer.py) or [`ssvad_metrics.visualize_dir`](ssvad_metrics/visualizer.py) for the Python API details and usage.
Requires FFMPEG installation on the system and `ffmpeg-python` package (**NOT** `python-ffmpeg`).
FFMPEG is used instead of OpenCV VideoWriter, since the OpenCV packages that are distributed in the PyPI usually does not embed FFMPEG that uis compiled with H264 codec.

## References

1. B. Ramachandra, M. Jones and R. R. Vatsavai, "A Survey of Single-Scene Video Anomaly Detection," in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2020.3040591.

## License

GPL-3.0 License. Brought to open-source by PT Qlue Performa Indonesia.

## Contributing

Feel free to contribute for improvements.
