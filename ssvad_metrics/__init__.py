"""
Single Scene Video Anomaly Detection Metrics
"""
import pkgutil

from ssvad_metrics import criteria, data_schema, metrics, utils, visualizer
from ssvad_metrics.metrics import accumulated_evaluate, evaluate
from ssvad_metrics.visualizer import visualize, visualize_dir

__version__ = pkgutil.get_data("ssvad_metrics", "VERSION").decode("utf-8")
