"""
YouTube Dataset to Audio Features Adapter

Converts YouTube dataset records into the format expected by the audio features training pipeline.
Handles the mapping between YouTube metadata and the audio features training data structure.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .types import (
    PerformanceMetrics,
    ProcessingConfig,
    TrainingDataPoint,
    VideoMetadata,
)


class YouTubeDatasetAdapter:
    """
    Adapter to convert YouTube dataset records to audio features training format.

    Handles:
    - Converting YouTube VideoRecordDict to TrainingDataPoint format
    - Performance label generation based on engagement metrics
    - Video file path resolution and validation
    - Metadata normalization and enrichment
    """

    def __init__(self, data_dir: str = "data", config: ProcessingConfig | None = None):
        """
        Initialize the YouTube dataset adapter.

        Args:
            data_dir: Directory containing YouTube dataset
            config: Processing configuration for audio features pipeline
        """
        self.data_dir = Path(data_dir)
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Common paths
        self.cache_dir = self.data_dir / "cache"
        self.video_cache_dir = self.cache_dir / "videos"
        self.metadata_cache_dir = self.cache_dir / "metadata"
        self.transcript_cache_dir = self.cache_dir / "transcripts"

    def load_youtube_dataset(self) -> list[dict[str, Any]]:
        """
        Load YouTube dataset from the collected brand data.

        Returns:
            List of YouTube video records
        """
        self.logger.info("Loading YouTube dataset...")

        records = []
        raw_dir = self.data_dir / "raw" / "brands"

        if not raw_dir.exists():
            self.logger.error(f"Raw data directory not found: {raw_dir}")
            return []

        # Load data from each brand directory
        for brand_dir in raw_dir.iterdir():
            if brand_dir.is_dir():
                brand_name = brand_dir.name
                self.logger.info(f"Loading data for brand: {brand_name}")

                # Load JSON files
                for json_file in brand_dir.glob("*.json"):
                    try:
                        with open(json_file, encoding="utf-8") as f:
                            brand_data = json.load(f)

                        if isinstance(brand_data, list):
                            for video in brand_data:
                                video["brand_name"] = brand_name
                                records.append(video)
                        else:
                            brand_data["brand_name"] = brand_name
                            records.append(brand_data)

                    except Exception as e:
                        self.logger.warning(f"Error loading {json_file}: {e}")

        self.logger.info(f"Loaded {len(records)} YouTube videos total")
        return records

    def _validate_video_record(self, record: dict[str, Any]) -> bool:
        """
        Validate that a YouTube record has required fields and accessible video file.

        Args:
            record: YouTube video record

        Returns:
            True if record is valid for training, False otherwise
        """
        # Check required fields
        required_fields = ["video_id", "view_count", "like_count", "comment_count"]
        for field in required_fields:
            if field not in record or record[field] is None:
                return False

        # Check if video file exists
        video_id = record.get("video_id", "")
        video_path = self._get_video_path(video_id)
        if not video_path or not video_path.exists():
            return False

        return True

    def _get_video_path(self, video_id: str) -> Path | None:
        """Get the path to the video file for a given video ID."""
        if not video_id:
            return None

        # Check in cache directory
        video_path = self.video_cache_dir / f"{video_id}.mp4"
        if video_path.exists():
            return video_path

        return None

    def _get_transcript_path(self, video_id: str) -> Path | None:
        """Get the path to the transcript file for a given video ID."""
        if not video_id:
            return None

        # Check in cache directory
        transcript_path = self.transcript_cache_dir / f"{video_id}.txt"
        if transcript_path.exists():
            return transcript_path

        return None

    def _load_transcript(self, video_id: str) -> str:
        """Load transcript text for a video."""
        transcript_path = self._get_transcript_path(video_id)
        if not transcript_path:
            return ""

        try:
            with open(transcript_path, encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            self.logger.warning(f"Error loading transcript for {video_id}: {e}")
            return ""

    def _create_performance_metrics(self, record: dict[str, Any]) -> PerformanceMetrics:
        """
        Create performance metrics from YouTube record.

        Args:
            record: YouTube video record

        Returns:
            PerformanceMetrics object
        """

        def safe_float(value, default=0.0):
            try:
                if isinstance(value, str):
                    return float(value.replace(",", ""))
                return float(value or 0)
            except (ValueError, TypeError):
                return default

        view_count = safe_float(record.get("view_count", 0))
        like_count = safe_float(record.get("like_count", 0))
        comment_count = safe_float(record.get("comment_count", 0))

        # Calculate engagement rate
        engagement_rate = 0.0
        if view_count > 0:
            engagement_rate = (like_count + comment_count) / view_count

        return PerformanceMetrics(
            view_count=int(view_count),
            like_count=int(like_count),
            dislike_count=0,  # YouTube API no longer provides dislike counts
            comment_count=int(comment_count),
            share_count=0,  # Not available in current dataset
            engagement_rate=engagement_rate,
            click_through_rate=0.0,  # Not available in current dataset
            conversion_rate=0.0,  # Not available in current dataset
            cost_per_click=0.0,  # Not available in current dataset
            return_on_ad_spend=0.0,  # Not available in current dataset
        )

    def _create_video_metadata(self, record: dict[str, Any]) -> VideoMetadata:
        """
        Create video metadata from YouTube record.

        Args:
            record: YouTube video record

        Returns:
            VideoMetadata object
        """
        video_id = record.get("video_id", "")
        video_path = self._get_video_path(video_id)

        # Get file size if video exists
        file_size_mb = 0.0
        if video_path and video_path.exists():
            try:
                file_size_mb = video_path.stat().st_size / (1024 * 1024)
            except Exception:
                pass

        def safe_float(value, default=0.0):
            try:
                return float(value or 0)
            except (ValueError, TypeError):
                return default

        return VideoMetadata(
            file_path=str(video_path) if video_path else "",
            file_size_mb=file_size_mb,
            duration_seconds=safe_float(record.get("duration_seconds", 0)),
            video_codec=record.get("video_codec", ""),
            audio_codec=record.get("audio_codec", ""),
            sample_rate=self.config.get("sample_rate", 22050),
            channels=2,  # Assume stereo
            bitrate=0,  # Not available in current dataset
            created_date=record.get("published_at", ""),
            modified_date=record.get("published_at", ""),
        )

    def _generate_performance_label(
        self, performance_metrics: PerformanceMetrics, record: dict[str, Any]
    ) -> str:
        """
        Generate binary performance label (high/low) based on engagement metrics.

        Uses a combination of engagement rate and view velocity (views per day per subscriber).

        Args:
            performance_metrics: Performance metrics for the video
            record: Original YouTube record for additional context

        Returns:
            "high" or "low" performance label
        """

        def safe_float(value, default=0.0):
            try:
                return float(value or 0)
            except (ValueError, TypeError):
                return default

        # Calculate engagement score
        engagement_rate = performance_metrics["engagement_rate"]

        # Calculate view velocity (views per day per 1k subscribers)
        views = safe_float(record.get("view_count", 0))
        subscribers = safe_float(record.get("channel_subscriber_count", 0))
        age_days = safe_float(record.get("publish_age_days", 1))
        age_days = max(1.0, min(30.0, age_days))  # Cap at 30 days

        views_per_day = views / age_days
        subscriber_normalized_velocity = 0.0
        if subscribers > 0:
            subscriber_normalized_velocity = views_per_day / (subscribers / 1000.0)

        # Combined score: weight engagement rate and normalized velocity
        combined_score = (
            0.6 * engagement_rate * 1000  # Scale engagement rate
            + 0.4
            * min(subscriber_normalized_velocity, 1000)  # Cap velocity contribution
        )

        # Use a threshold based on typical performance
        # High performance: top 25% of content (this could be tuned based on dataset analysis)
        threshold = 15.0  # Adjust based on dataset characteristics

        return "high" if combined_score >= threshold else "low"

    def convert_to_training_data_points(
        self, youtube_records: list[dict[str, Any]]
    ) -> list[TrainingDataPoint]:
        """
        Convert YouTube records to TrainingDataPoint format for audio features training.

        Args:
            youtube_records: List of YouTube video records

        Returns:
            List of TrainingDataPoint objects ready for audio features training
        """
        self.logger.info("Converting YouTube records to training data points...")

        training_points = []
        processed_count = 0
        skipped_count = 0

        for record in youtube_records:
            try:
                # Validate record
                if not self._validate_video_record(record):
                    skipped_count += 1
                    continue

                video_id = record["video_id"]

                # Create performance metrics
                performance_metrics = self._create_performance_metrics(record)

                # Generate performance label
                performance_label = self._generate_performance_label(
                    performance_metrics, record
                )

                # Create video metadata
                video_metadata = self._create_video_metadata(record)

                # Create training data point (features will be extracted by the pipeline)
                training_point = TrainingDataPoint(
                    video_id=video_id,
                    campaign_id=record.get(
                        "brand_name", "unknown"
                    ),  # Use brand as campaign
                    brand_name=record.get("brand_name", "unknown"),
                    audio_features={},  # Will be populated by audio feature extractor
                    transcript_features={},  # Will be populated by transcript feature extractor
                    video_metadata=video_metadata,
                    performance_metrics=performance_metrics,
                    performance_label=performance_label,
                    extracted_date=datetime.now().isoformat(),
                    feature_version="1.0",
                    processing_time_seconds=0.0,  # Will be updated during processing
                )

                training_points.append(training_point)
                processed_count += 1

                if processed_count % 50 == 0:
                    self.logger.info(f"Processed {processed_count} records...")

            except Exception as e:
                self.logger.warning(
                    f"Error processing record {record.get('video_id', 'unknown')}: {e}"
                )
                skipped_count += 1

        self.logger.info(
            f"Conversion complete: {processed_count} training points created, {skipped_count} skipped"
        )
        return training_points

    def save_training_data(
        self, training_points: list[TrainingDataPoint], output_path: str
    ):
        """
        Save training data points to JSON file.

        Args:
            training_points: List of training data points
            output_path: Path to save the training data
        """
        self.logger.info(
            f"Saving {len(training_points)} training data points to {output_path}"
        )

        # Convert to serializable format
        serializable_data = []
        for point in training_points:
            # Convert to dict and ensure all values are JSON serializable
            point_dict = dict(point)
            serializable_data.append(point_dict)

        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2, default=str)

        self.logger.info(f"Training data saved to {output_path}")

    def get_dataset_statistics(
        self, training_points: list[TrainingDataPoint]
    ) -> dict[str, Any]:
        """
        Generate statistics about the converted training dataset.

        Args:
            training_points: List of training data points

        Returns:
            Dictionary with dataset statistics
        """
        if not training_points:
            return {}

        # Label distribution
        high_count = sum(1 for p in training_points if p["performance_label"] == "high")
        low_count = len(training_points) - high_count

        # Brand distribution
        brand_counts = {}
        for point in training_points:
            brand = point["brand_name"]
            brand_counts[brand] = brand_counts.get(brand, 0) + 1

        # Performance metrics statistics
        engagement_rates = [
            p["performance_metrics"]["engagement_rate"] for p in training_points
        ]
        view_counts = [p["performance_metrics"]["view_count"] for p in training_points]

        stats = {
            "total_videos": len(training_points),
            "label_distribution": {
                "high": high_count,
                "low": low_count,
                "high_percentage": (high_count / len(training_points)) * 100,
            },
            "brand_distribution": brand_counts,
            "engagement_rate_stats": {
                "mean": np.mean(engagement_rates),
                "median": np.median(engagement_rates),
                "std": np.std(engagement_rates),
                "min": np.min(engagement_rates),
                "max": np.max(engagement_rates),
            },
            "view_count_stats": {
                "mean": np.mean(view_counts),
                "median": np.median(view_counts),
                "std": np.std(view_counts),
                "min": np.min(view_counts),
                "max": np.max(view_counts),
            },
        }

        return stats
