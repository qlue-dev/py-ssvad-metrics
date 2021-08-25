# Single-Scene Video Anomaly Detection Metrics

This project contains evaluation protocol (metrics) for benchmarking single-scene video anomaly detection.

## Evaluation Protocol

This is an unofficial implementation of Sec. 2.2 of [A Survey of Single-Scene Video Anomaly Detection](https://arxiv.org/pdf/2004.05993.pdf).

## Installation

This metrics is available via PyPI.

```bash
pip install py-ssvad-metrics
```

## Usage

1. Prepare ground-truth JSON file and prediction JSON file. Examples are in the `tests` folder.
1. For UCSD Pedestrian 1 and 2 datasets, CUHK Avenue dataset, and Street Scene dataset,
we provided scripts for converting ground-truth annotation files from Street Scene dataset. Download link is provided in the paper [http://www.merl.com/demos/video-anomaly-detection].
1. Example usage for single groundtruth and prediction file pair:

    ```python
    import ssvad_metrics
    result = ssvad_metrics.metrics.evaluate(
        "tests/gt_examples/Test001_gt.json",
        "tests/pred_examples/Test001_pred.json")
    ```

1. Example usage for multiple groundtruth and prediction file pairs:

    ```python
    import ssvad_metrics
    result = ssvad_metrics.metrics.accumulated_evaluate(
        "tests/gt_examples",
        "tests/pred_examples",
        gt_name_suffix="_gt",
        pred_name_suffix="_pred")
    ```

## References

1. B. Ramachandra, M. Jones and R. R. Vatsavai, "A Survey of Single-Scene Video Anomaly Detection," in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2020.3040591.

## Copyright

Copyright PT Qlue Performa Indonesia 2021 All Rights Reserved

## Contributing

Feel free to contribute for improvements.
