"""
Type definitions for the Audio Features Advertisement Performance System.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


# Audio Analysis Types - Streamlined to 10 Most Important Features
class AudioFeatures(TypedDict, total=False):
    """Streamlined audio features - top 10 most predictive for advertisement performance."""

    # Energy & Engagement (most critical for ad impact)
    rms_mean: float  # Overall energy/loudness - key for attention
    dynamic_range: float  # Energy variation - indicates excitement/drama

    # Voice Quality & Clarity (critical for message delivery)
    pitch_mean: float  # Voice pitch - affects trust and authority
    speech_rate: float  # Speaking pace - affects comprehension

    # Spectral Characteristics (audio quality perception)
    spectral_centroid_mean: float  # Brightness - affects perceived quality
    mfcc_1_mean: float  # Primary speech characteristic
    mfcc_2_mean: float  # Secondary speech characteristic

    # Temporal Flow (engagement and retention)
    pause_duration_mean: float  # Speaking rhythm - affects flow
    onset_rate: float  # Event density - indicates activity level
    zero_crossing_rate: float  # Speech vs music distinction


class TranscriptFeatures(TypedDict, total=False):
    """Streamlined transcript features - top 10 most predictive for advertisement performance."""

    # Hook Analysis (Critical first 3-5 seconds for retention)
    hook_curiosity_words_count: int  # "Secret", "shocking" - drives initial interest
    hook_action_words_count: int  # "Get", "try", "buy" - immediate CTAs
    hook_personal_pronouns_count: int  # "You", "your" - personal connection
    hook_sentiment_polarity: float  # Positive emotion in opening

    # Core Marketing Elements (conversion drivers)
    call_to_action_count: int  # "Buy now", "click here" - direct conversion
    action_words_count: int  # Total action-oriented language
    sentiment_polarity: float  # Overall emotional tone

    # Engagement Indicators (retention factors)
    exclamation_count: int  # Energy and excitement level
    question_count: int  # Audience engagement techniques
    word_count: int  # Content density and pacing


class VideoMetadata(TypedDict, total=False):
    """Metadata about the video file."""

    file_path: str
    file_size_mb: float
    duration_seconds: float
    video_codec: str
    audio_codec: str
    sample_rate: int
    channels: int
    bitrate: int
    created_date: str
    modified_date: str


class PerformanceMetrics(TypedDict):
    """Performance metrics for the advertisement."""

    view_count: int
    like_count: int
    dislike_count: int
    comment_count: int
    share_count: int
    engagement_rate: float
    click_through_rate: float
    conversion_rate: float
    cost_per_click: float
    return_on_ad_spend: float


class TrainingDataPoint(TypedDict):
    """Complete training data point combining all features and labels."""

    # Identifiers
    video_id: str
    campaign_id: str
    brand_name: str

    # Features
    audio_features: AudioFeatures
    transcript_features: TranscriptFeatures
    video_metadata: VideoMetadata

    # Labels
    performance_metrics: PerformanceMetrics
    performance_label: Literal["high", "low"]  # Binary classification target

    # Processing metadata
    extracted_date: str
    feature_version: str
    processing_time_seconds: float


class ModelPrediction(TypedDict):
    """Prediction output from the models."""

    # Audio model predictions
    audio_model_score: float
    audio_model_confidence: float

    # Transcript model predictions
    transcript_model_score: float
    transcript_model_confidence: float

    # Fusion model predictions
    fusion_model_score: float
    fusion_model_confidence: float
    final_prediction: Literal["high", "low"]

    # Feature contributions
    top_audio_features: list[tuple[str, float]]
    top_transcript_features: list[tuple[str, float]]

    # Processing info
    processing_time_seconds: float
    model_version: str
    prediction_date: str


class ProcessingConfig(TypedDict, total=False):
    """Configuration for audio processing pipeline."""

    # Audio processing
    sample_rate: int  # Target sample rate (default: 22050)
    frame_length: int  # FFT window size (default: 2048)
    hop_length: int  # Hop size for windowing (default: 512)

    # Feature extraction
    n_mfcc: int  # Number of MFCC coefficients (default: 13)
    n_fft: int  # FFT window size for spectral features

    # Speech recognition
    asr_service: Literal["whisper", "google", "azure", "aws"]
    asr_language: str  # Language code (default: "en-US")
    asr_confidence_threshold: float

    # Text processing
    remove_stopwords: bool
    stem_words: bool
    max_text_length: int


class AudioProcessingResult(TypedDict):
    """Result of audio processing operations."""

    success: bool
    audio_file_path: str
    sample_rate: int
    duration_seconds: float
    channels: int
    features_extracted: bool
    transcript_extracted: bool
    error_message: str | None
    processing_time_seconds: float


class BatchProcessingJob(TypedDict):
    """Configuration for batch processing jobs."""

    job_id: str
    input_directory: str
    output_directory: str
    file_patterns: list[str]
    processing_config: ProcessingConfig
    parallel_workers: int
    status: Literal["pending", "running", "completed", "failed"]
    progress: float  # 0.0 to 1.0
    start_time: str
    end_time: str | None
    files_processed: int
    files_failed: int
    total_files: int


class ModelTrainingConfig(TypedDict):
    """Configuration for model training."""

    # Data
    train_data_path: str
    validation_split: float
    test_split: float

    # Model parameters
    model_type: Literal["xgboost", "lightgbm", "random_forest", "neural_network"]
    hyperparameters: dict[str, Any]

    # Training
    cross_validation_folds: int
    random_state: int
    early_stopping_rounds: int

    # Output
    model_save_path: str
    experiment_name: str
    track_experiments: bool


class ModelPerformanceMetrics(TypedDict):
    """Metrics for evaluating model performance."""

    # Classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    auc_pr: float

    # Confusion matrix
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    # Cross-validation
    cv_mean: float
    cv_std: float

    # Feature importance
    feature_importance: dict[str, float]

    # Training info
    training_time_seconds: float
    model_size_mb: float
    training_samples: int
    validation_samples: int


# API Types
class PredictionRequest(TypedDict):
    """Request payload for prediction API."""

    video_file: bytes | str  # Binary data or file path
    processing_config: ProcessingConfig | None
    return_features: bool
    return_explanations: bool


class PredictionResponse(TypedDict):
    """Response from prediction API."""

    prediction: ModelPrediction
    features: dict[str, Any] | None  # Optional detailed features
    explanations: dict[str, str] | None  # Optional explanations
    processing_time_ms: int
    api_version: str
    request_id: str
