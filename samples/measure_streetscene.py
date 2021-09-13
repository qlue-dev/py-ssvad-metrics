import os

import pandas as pd
import ssvad_metrics

if __name__ == "__main__":

    gt_dir_root = "vad_datasets/py-ssvad-metrics-formatted/"
    pred_dir_root = "prediction/"

    preds = [
        "streetscene/method_a",
        "streetscene/method_b",
        "streetscene/method_c"]

    results = [
        {
            "name": n,
            **ssvad_metrics.metrics.accumulated_evaluate(
                os.path.join(gt_dir_root, "streetscene-halfscale"),
                os.path.join(pred_dir_root, n),
                gt_name_suffix="_gt",
                pred_name_suffix="_pred")}
        for n in preds]
    df = pd.DataFrame(results)
    df.to_csv("streetscene-halfscale_summary.csv")
