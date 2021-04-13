"""
Single Scene Video Anomaly Detection Metrics
"""
import pkgutil

from ssvad_metrics import criteria, data_schema, metrics, utils

__version__ = pkgutil.get_data("ssvad_metrics", "VERSION").decode("utf-8")
