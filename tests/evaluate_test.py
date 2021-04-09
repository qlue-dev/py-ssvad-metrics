import ssvad_metrics

def test_evaluate_gt_gt():
    result = ssvad_metrics.metrics.evaluate(
        "tests/Test001_gt.json",
        "tests/Test001_gt.json")
    print(result)

if __name__ == "__main__":
    test_evaluate_gt_gt()