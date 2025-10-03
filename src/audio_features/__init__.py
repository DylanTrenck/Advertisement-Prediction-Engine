"""
Audio Features Advertisement Performance System

A modular system for predicting advertisement performance using audio analysis,
speech recognition, and text analysis.
"""

from .audio_extractor import AudioExtractor
from .audio_features import AudioFeatureExtractor
from .fusion_model import FusionModel
from .main_pipeline import AudioAdPerformanceModel
from .transcript_extractor import TranscriptExtractor
from .transcript_features import TranscriptFeatureExtractor
from .types import (
    AudioFeatures,
    AudioProcessingResult,
    BatchProcessingJob,
    ModelPerformanceMetrics,
    ModelPrediction,
    ModelTrainingConfig,
    PerformanceMetrics,
    PredictionRequest,
    PredictionResponse,
    ProcessingConfig,
    TrainingDataPoint,
    TranscriptFeatures,
    VideoMetadata,
)
from .youtube_adapter import YouTubeDatasetAdapter

# Version info
__version__ = "0.1.0"
__author__ = "AudioAd Team"

# Main public API
__all__ = [
    # Core pipeline
    "AudioAdPerformanceModel",
    # Individual components
    "AudioExtractor",
    "AudioFeatureExtractor",
    "TranscriptExtractor",
    "TranscriptFeatureExtractor",
    "FusionModel",
    "YouTubeDatasetAdapter",
    # Type definitions
    "AudioFeatures",
    "TranscriptFeatures",
    "VideoMetadata",
    "PerformanceMetrics",
    "TrainingDataPoint",
    "ModelPrediction",
    "ProcessingConfig",
    "AudioProcessingResult",
    "BatchProcessingJob",
    "ModelTrainingConfig",
    "ModelPerformanceMetrics",
    "PredictionRequest",
    "PredictionResponse",
    # Package info
    "__version__",
    "__author__",
]
