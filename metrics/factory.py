"""Metric Factory Function for creating metrics."""

# factory.py

from enum import Enum

from metrics.editing_score import EditingScore
from .psnr import PSNR
from .ssim import SSIM
from .fsim import FSIM
from .lpips_score import LipisScore
from metrics.segmentation_score import SegmentationScore


class MetricType(Enum):
    PSNR    = "PSNR"
    SSIM    = "SSIM"
    FSIM    = "FSIM"
    MASKED  = "MASKED"
    QWEN    = "QWEN"
    SEG     = "SEG"


def create_metric(metric_type: MetricType, **kwargs):
    if metric_type == MetricType.PSNR:
        return PSNR()
    elif metric_type == MetricType.SSIM:
        return SSIM()
    elif metric_type == MetricType.FSIM:
        return FSIM()
    elif metric_type == MetricType.MASKED:
        return LipisScore(**kwargs)
    elif metric_type == MetricType.QWEN:
        return EditingScore(**kwargs)
    elif metric_type == MetricType.SEG:
        return SegmentationScore(**kwargs)  
    else:
        raise ValueError(f"Invalid metric name: {metric_type}")