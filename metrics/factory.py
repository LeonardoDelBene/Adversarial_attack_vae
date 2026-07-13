"""Metric Factory Function for creating metrics."""

# factory.py

from enum import Enum

from metrics.editing_score import EditingScore
from .psnr import PSNR
from .ssim import SSIM
from .clip_score import ClipScore
from .fsim import FSIM
from .caption_similarity import CaptionSimilarity, AccuracyRate
from .lpips_score import LipisScore


class MetricType(Enum):
    PSNR    = "PSNR"
    SSIM    = "SSIM"
    CLIP    = "CLIP"
    FSIM    = "FSIM"
    CAP     = "CAP"
    ACC     = "ACC"
    CLIP_DIR = "CLIP_DIR"
    MASKED  = "MASKED"
    QWEN    = "QWEN"


def create_metric(metric_type: MetricType, **kwargs):
    if metric_type == MetricType.PSNR:
        return PSNR()
    elif metric_type == MetricType.SSIM:
        return SSIM()
    elif metric_type == MetricType.CLIP:
        return ClipScore(**kwargs)
    elif metric_type == MetricType.FSIM:
        return FSIM()
    elif metric_type == MetricType.CAP:
        return CaptionSimilarity(**kwargs)
    elif metric_type == MetricType.ACC:
        return AccuracyRate(**kwargs)
    elif metric_type == MetricType.MASKED:
        return LipisScore(**kwargs)
    elif metric_type == MetricType.QWEN:
        return EditingScore(**kwargs)
    else:
        raise ValueError(f"Invalid metric name: {metric_type}")