"""
Main pipeline that orchestrates the complete audio advertisement performance prediction system.
"""

import asyncio
import logging
import time

from .audio_extractor import AudioExtractor
from .audio_features import AudioFeatureExtractor
from .fusion_model import FusionModel
from .transcript_extractor import TranscriptExtractor
from .transcript_features import TranscriptFeatureExtractor
from .types import (
    ModelPrediction,
    ProcessingConfig,
)


class AudioAdPerformanceModel:
    """
    Complete pipeline for predicting advertisement performance from MP4 files.

    Pipeline stages:
    1. Extract audio from MP4 → AudioExtractor
    2. Extract audio features → AudioFeatureExtractor
    3. Extract transcript → TranscriptExtractor
    4. Extract transcript features → TranscriptFeatureExtractor
    5. Combine features and predict → FusionModel

    Examples:
        >>> model = AudioAdPerformanceModel()
        >>> await model.load_models("path/to/models")
        >>> prediction = await model.predict_performance("ad_video.mp4")
        >>> assert prediction["final_prediction"] in ["high", "low"]
    """

    def __init__(self, config: ProcessingConfig | None = None):
        """
        Initialize the complete pipeline.

        Args:
            config: Processing configuration for all components
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or ProcessingConfig()

        # Initialize all components
        self.audio_extractor = AudioExtractor()
        self.audio_feature_extractor = AudioFeatureExtractor(config)
        self.transcript_extractor = TranscriptExtractor(config)
        self.transcript_feature_extractor = TranscriptFeatureExtractor(config)
        self.fusion_model = FusionModel()

        # State tracking
        self.models_loaded = False
        self.processing_stats = {
            "total_processed": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "avg_processing_time": 0.0,
        }

    async def predict_performance(self, video_path: str) -> ModelPrediction:
        """
        Predict advertisement performance from MP4 video file.

        Args:
            video_path: Path to the MP4 video file

        Returns:
            Prediction results with confidence scores and explanations
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        start_time = time.time()

        try:
            self.logger.info(f"Starting prediction pipeline for {video_path}")

            # Stage 1: Extract audio from MP4
            self.logger.info("Stage 1: Extracting audio from MP4...")
            audio_result = self.audio_extractor.extract_audio_from_mp4(
                video_path, config=self.config
            )

            if not audio_result["success"]:
                raise Exception(
                    f"Audio extraction failed: {audio_result['error_message']}"
                )

            audio_path = audio_result["audio_file_path"]

            # Stage 2: Extract audio features
            self.logger.info("Stage 2: Extracting audio features...")
            audio_features = self.audio_feature_extractor.extract_features(audio_path)

            # Stage 3: Extract transcript
            self.logger.info("Stage 3: Extracting transcript...")
            transcript = self.transcript_extractor.extract_transcript(audio_path)

            # Stage 4: Extract transcript features
            self.logger.info("Stage 4: Extracting transcript features...")
            transcript_features = self.transcript_feature_extractor.extract_features(
                transcript
            )

            # Stage 5: Combine and predict
            self.logger.info("Stage 5: Making final prediction...")
            prediction = self.fusion_model.predict(audio_features, transcript_features)

            # Update statistics
            self.processing_stats["total_processed"] += 1
            self.processing_stats["successful_predictions"] += 1

            processing_time = time.time() - start_time
            self._update_avg_processing_time(processing_time)

            self.logger.info(
                f"Prediction completed in {processing_time:.2f}s: "
                f"{prediction['final_prediction']} (confidence: {prediction['fusion_model_confidence']:.3f})"
            )

            return prediction

        except Exception as e:
            self.processing_stats["total_processed"] += 1
            self.processing_stats["failed_predictions"] += 1

            error_msg = f"Prediction failed for {video_path}: {str(e)}"
            self.logger.error(error_msg)

            # Return error prediction
            return ModelPrediction(
                audio_model_score=0.5,
                audio_model_confidence=0.0,
                transcript_model_score=0.5,
                transcript_model_confidence=0.0,
                fusion_model_score=0.5,
                fusion_model_confidence=0.0,
                final_prediction="low",
                top_audio_features=[],
                top_transcript_features=[],
                processing_time_seconds=time.time() - start_time,
                model_version="error",
                prediction_date=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

    async def predict_performance_batch(
        self, video_paths: list[str]
    ) -> list[ModelPrediction]:
        """
        Predict performance for multiple videos in parallel.

        Args:
            video_paths: List of paths to MP4 video files

        Returns:
            List of prediction results
        """
        self.logger.info(f"Starting batch prediction for {len(video_paths)} videos")

        # Process videos in parallel (with reasonable concurrency limit)
        semaphore = asyncio.Semaphore(4)  # Limit to 4 concurrent processes

        async def predict_with_semaphore(video_path: str) -> ModelPrediction:
            async with semaphore:
                return await self.predict_performance(video_path)

        # Create tasks for all videos
        tasks = [predict_with_semaphore(path) for path in video_paths]

        # Wait for all predictions to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        predictions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(
                    f"Batch prediction failed for {video_paths[i]}: {result}"
                )
                # Create error prediction
                predictions.append(self._create_error_prediction())
            else:
                predictions.append(result)

        return predictions

    def extract_features_only(self, video_path: str) -> dict:
        """
        Extract only features without making prediction (useful for training data).

        Args:
            video_path: Path to the MP4 video file

        Returns:
            Dictionary containing audio and transcript features
        """
        try:
            # Extract audio
            audio_result = self.audio_extractor.extract_audio_from_mp4(
                video_path, config=self.config
            )

            if not audio_result["success"]:
                raise Exception(
                    f"Audio extraction failed: {audio_result['error_message']}"
                )

            audio_path = audio_result["audio_file_path"]

            # Extract features
            audio_features = self.audio_feature_extractor.extract_features(audio_path)
            transcript = self.transcript_extractor.extract_transcript(audio_path)
            transcript_features = self.transcript_feature_extractor.extract_features(
                transcript
            )

            return {
                "audio_features": audio_features,
                "transcript_features": transcript_features,
                "transcript_text": transcript,
                "video_metadata": self.audio_extractor.get_video_info(video_path),
            }

        except Exception as e:
            self.logger.error(f"Feature extraction failed for {video_path}: {e}")
            return {
                "audio_features": {},
                "transcript_features": {},
                "transcript_text": "",
                "video_metadata": {},
                "error": str(e),
            }

    def load_models(self, model_dir: str) -> None:
        """
        Load trained models from directory.

        Args:
            model_dir: Directory containing saved model files
        """
        try:
            self.fusion_model.load_models(model_dir)
            self.models_loaded = True
            self.logger.info(f"Models loaded successfully from {model_dir}")
        except Exception as e:
            self.logger.error(f"Failed to load models from {model_dir}: {e}")
            raise

    def save_models(self, model_dir: str) -> None:
        """
        Save trained models to directory.

        Args:
            model_dir: Directory to save model files
        """
        try:
            self.fusion_model.save_models(model_dir)
            self.logger.info(f"Models saved successfully to {model_dir}")
        except Exception as e:
            self.logger.error(f"Failed to save models to {model_dir}: {e}")
            raise

    def get_processing_stats(self) -> dict:
        """Get processing statistics."""
        return self.processing_stats.copy()

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            "total_processed": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "avg_processing_time": 0.0,
        }

    def _update_avg_processing_time(self, new_time: float) -> None:
        """Update running average of processing time."""
        total = self.processing_stats["successful_predictions"]
        current_avg = self.processing_stats["avg_processing_time"]

        # Calculate new average
        new_avg = ((current_avg * (total - 1)) + new_time) / total
        self.processing_stats["avg_processing_time"] = new_avg

    def _create_error_prediction(self) -> ModelPrediction:
        """Create a default error prediction."""
        return ModelPrediction(
            audio_model_score=0.5,
            audio_model_confidence=0.0,
            transcript_model_score=0.5,
            transcript_model_confidence=0.0,
            fusion_model_score=0.5,
            fusion_model_confidence=0.0,
            final_prediction="low",
            top_audio_features=[],
            top_transcript_features=[],
            processing_time_seconds=0.0,
            model_version="error",
            prediction_date=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def cleanup(self) -> None:
        """Clean up temporary files and resources."""
        try:
            # Clean up temporary audio files
            self.audio_extractor.cleanup_temp_files()
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.warning(f"Cleanup warning: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


# Convenience function for simple usage
async def predict_ad_performance(
    video_path: str, model_dir: str, config: ProcessingConfig | None = None
) -> ModelPrediction:
    """
    Convenience function to predict ad performance from a single video.

    Args:
        video_path: Path to the MP4 video file
        model_dir: Directory containing trained models
        config: Optional processing configuration

    Returns:
        Prediction results

    Examples:
        >>> prediction = await predict_ad_performance("ad.mp4", "models/")
        >>> print(f"Prediction: {prediction['final_prediction']}")
    """
    with AudioAdPerformanceModel(config) as model:
        model.load_models(model_dir)
        return await model.predict_performance(video_path)
