#!/usr/bin/env python3
"""
Content-Based YouTube Training Script (313 Videos)

Train on the exact same 313 videos that achieved 96.8% metadata accuracy,
but using audio and transcript content analysis instead.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import ffmpeg
import joblib

# Audio and text processing
import librosa
import nltk
import numpy as np
import pandas as pd
import speech_recognition as sr
import xgboost as xgb
from pydub import AudioSegment
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("content_based_313_training.log"),
        ],
    )


class ContentBased313Trainer:
    """
    Content-based trainer using the exact same 313 videos as the metadata model.
    """

    def __init__(self, data_dir: str = "../../data"):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)

        # Directories (same as simple trainer)
        self.cache_dir = self.data_dir / "cache"
        self.video_cache_dir = self.cache_dir / "videos"
        self.transcript_cache_dir = self.cache_dir / "transcripts"
        self.raw_dir = self.data_dir / "raw" / "brands"

        # Create output directories
        self.models_dir = Path("../models")
        self.results_dir = Path("../results")
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        # Initialize extractors
        self.recognizer = sr.Recognizer()

        # Download NLTK data
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
        except:
            pass

        # Temporary directory for audio files
        self.temp_dir = tempfile.mkdtemp()

        # Statistics
        self.stats = {
            "total_videos_loaded": 0,
            "videos_with_files": 0,
            "audio_extracted": 0,
            "transcripts_extracted": 0,
            "training_samples": 0,
            "high_performance": 0,
            "low_performance": 0,
        }

    def load_youtube_data(self) -> list[dict[str, Any]]:
        """Load YouTube dataset from JSON files (same logic as simple trainer)."""
        self.logger.info("Loading YouTube dataset...")

        records = []

        if not self.raw_dir.exists():
            self.logger.error(f"Raw data directory not found: {self.raw_dir}")
            return []

        # Load data from each brand directory
        for brand_dir in self.raw_dir.iterdir():
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

        self.stats["total_videos_loaded"] = len(records)
        self.logger.info(f"Loaded {len(records)} YouTube videos total")
        return records

    def create_performance_label(self, record: dict[str, Any]) -> str:
        """Create binary performance label (same logic as simple trainer)."""

        def safe_float(value, default=0.0):
            try:
                if isinstance(value, str):
                    return float(value.replace(",", ""))
                return float(value or 0)
            except (ValueError, TypeError):
                return default

        views = safe_float(record.get("view_count", 0))
        likes = safe_float(record.get("like_count", 0))
        comments = safe_float(record.get("comment_count", 0))
        subscribers = safe_float(record.get("channel_subscriber_count", 0))
        age_days = safe_float(record.get("publish_age_days", 1))
        age_days = max(1.0, min(30.0, age_days))  # Cap at 30 days

        # Calculate engagement rate
        engagement_rate = 0.0
        if views > 0:
            engagement_rate = (likes + comments) / views

        # Calculate view velocity (views per day)
        views_per_day = views / age_days

        # Normalize by subscriber count if available
        if subscribers > 0:
            subscriber_normalized_velocity = views_per_day / (subscribers / 1000.0)
        else:
            subscriber_normalized_velocity = (
                views_per_day / 1000.0
            )  # Assume 1k subscribers

        # Combined score (same threshold as simple trainer)
        combined_score = (
            0.6 * engagement_rate * 1000  # Scale engagement rate
            + 0.4 * min(subscriber_normalized_velocity, 1000)  # Cap velocity
        )

        # Threshold for high performance (same as simple trainer)
        threshold = 15.0

        return "high" if combined_score >= threshold else "low"

    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video file."""
        try:
            audio_output_path = os.path.join(
                self.temp_dir, f"temp_audio_{os.getpid()}.wav"
            )

            # Use ffmpeg to extract audio
            (
                ffmpeg.input(video_path)
                .output(audio_output_path, acodec="pcm_s16le", ac=1, ar="16000")
                .overwrite_output()
                .run(quiet=True)
            )

            return audio_output_path

        except Exception as e:
            self.logger.error(f"Audio extraction failed: {e}")
            raise

    def extract_audio_features(self, audio_path: str) -> dict[str, float]:
        """Extract 10 critical audio features using librosa."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050)

            if len(y) == 0:
                return self._get_empty_audio_features()

            features = {}

            # 1. RMS Energy (attention-grabbing quality)
            rms = librosa.feature.rms(y=y)[0]
            features["audio_rms_mean"] = float(np.mean(rms))

            # 2. Dynamic Range (energy variation)
            features["audio_dynamic_range"] = float(np.max(rms) - np.min(rms))

            # 3. Spectral Centroid (brightness/quality perception)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features["audio_spectral_centroid_mean"] = float(np.mean(centroid))

            # 4-5. MFCC Features (speech characteristics)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features["audio_mfcc_1_mean"] = float(np.mean(mfccs[0]))
            features["audio_mfcc_2_mean"] = float(np.mean(mfccs[1]))

            # 6. Pitch/Fundamental Frequency
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if pitch_values:
                features["audio_pitch_mean"] = float(np.mean(pitch_values))
            else:
                features["audio_pitch_mean"] = 0.0

            # 7. Speech Rate (onset detection)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            duration = len(y) / sr
            if duration > 0:
                features["audio_speech_rate"] = float(len(onset_frames) / duration)
            else:
                features["audio_speech_rate"] = 0.0

            # 8. Pause Duration (silence detection)
            rms_threshold = np.percentile(rms, 20)  # Bottom 20% as silence
            silent_frames = rms < rms_threshold

            # Find silent segments
            silent_segments = []
            in_silence = False
            silence_start = 0

            for i, is_silent in enumerate(silent_frames):
                if is_silent and not in_silence:
                    silence_start = i
                    in_silence = True
                elif not is_silent and in_silence:
                    silent_segments.append(i - silence_start)
                    in_silence = False

            if silent_segments:
                # Convert frames to seconds
                frame_duration = len(y) / len(rms) / sr
                pause_durations = [seg * frame_duration for seg in silent_segments]
                features["audio_pause_duration_mean"] = float(np.mean(pause_durations))
            else:
                features["audio_pause_duration_mean"] = 0.0

            # 9. Onset Rate (rhythm/engagement)
            if duration > 0:
                features["audio_onset_rate"] = float(len(onset_frames) / duration)
            else:
                features["audio_onset_rate"] = 0.0

            # 10. Zero Crossing Rate (voice activity)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features["audio_zero_crossing_rate"] = float(np.mean(zcr))

            return features

        except Exception as e:
            self.logger.error(f"Audio feature extraction failed: {e}")
            return self._get_empty_audio_features()

    def _get_empty_audio_features(self) -> dict[str, float]:
        """Return empty audio features dict."""
        return {
            "audio_rms_mean": 0.0,
            "audio_dynamic_range": 0.0,
            "audio_spectral_centroid_mean": 0.0,
            "audio_mfcc_1_mean": 0.0,
            "audio_mfcc_2_mean": 0.0,
            "audio_pitch_mean": 0.0,
            "audio_speech_rate": 0.0,
            "audio_pause_duration_mean": 0.0,
            "audio_onset_rate": 0.0,
            "audio_zero_crossing_rate": 0.0,
        }

    def extract_transcript(self, audio_path: str) -> str:
        """
        More robust transcription:
        - normalize audio
        - trim silence
        - chunk by silence and by max duration
        - retries per chunk
        """
        import tempfile

        from pydub import AudioSegment, effects
        from pydub.silence import split_on_silence

        try:
            # Load and normalize audio
            seg = (
                AudioSegment.from_wav(audio_path).set_channels(1).set_frame_rate(16000)
            )
            seg = effects.normalize(seg)

            # Trim leading/trailing silence
            seg = seg.strip_silence(
                silence_len=300, silence_thresh=seg.dBFS - 20, padding=100
            )

            # Split on silence into reasonable phrases
            chunks = split_on_silence(
                seg,
                min_silence_len=400,
                silence_thresh=seg.dBFS - 20,
                keep_silence=150,
            )

            # Fallback: if no good silence splits, use fixed windows
            if not chunks:
                step_ms = 25_000  # 25s
                chunks = [seg[i : i + step_ms] for i in range(0, len(seg), step_ms)]

            transcripts = []
            tmpdir = tempfile.mkdtemp()

            # Configure recognizer for better performance
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.energy_threshold = 300  # starting point

            for _i, chunk in enumerate(chunks):
                # Cap chunk length to 30s
                if len(chunk) > 30_000:
                    for j in range(0, len(chunk), 30_000):
                        sub = chunk[j : j + 30_000]
                        text = self._transcribe_chunk(sub, tmpdir)
                        if text:
                            transcripts.append(text)
                else:
                    text = self._transcribe_chunk(chunk, tmpdir)
                    if text:
                        transcripts.append(text)

            # Clean up temp directory
            try:
                import shutil

                shutil.rmtree(tmpdir)
            except:
                pass

            return " ".join(transcripts).strip()

        except Exception as e:
            self.logger.error(f"Transcript extraction failed: {e}")
            return ""

    def _transcribe_chunk(self, chunk: AudioSegment, tmpdir: str) -> str:
        """Transcribe a single audio chunk with retries."""
        import os
        import uuid

        path = os.path.join(tmpdir, f"{uuid.uuid4().hex}.wav")
        try:
            # Export chunk as WAV
            chunk.export(
                path,
                format="wav",
                parameters=["-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000"],
            )

            with sr.AudioFile(path) as source:
                audio = self.recognizer.record(source)

            # Try recognition with language hint and retries
            for attempt in range(2):
                try:
                    return self.recognizer.recognize_google(audio, language="en-US")
                except sr.UnknownValueError:
                    if attempt == 0:
                        # Try adjusting energy threshold for second attempt
                        self.recognizer.energy_threshold = max(
                            50, self.recognizer.energy_threshold - 100
                        )
                        continue
                    return ""
                except sr.RequestError as e:
                    self.logger.warning(f"Google SR request error: {e}")
                    return ""
                except Exception as e:
                    self.logger.warning(f"Unexpected SR error: {e}")
                    return ""

        finally:
            try:
                os.remove(path)
            except:
                pass

    def extract_transcript_features(self, transcript: str) -> dict[str, float]:
        """Extract 10 critical transcript features."""
        if not transcript or not transcript.strip():
            return self._get_empty_transcript_features()

        features = {}
        text = transcript.lower()
        words = text.split()

        # Hook Analysis Features (first 15 words)
        hook_text = " ".join(words[:15]) if len(words) >= 15 else text

        # 1. Hook Curiosity Words
        curiosity_words = [
            "discover",
            "secret",
            "revealed",
            "hidden",
            "mystery",
            "unknown",
            "surprising",
            "shocking",
            "incredible",
            "amazing",
            "unbelievable",
            "what",
            "how",
            "why",
            "when",
            "where",
        ]
        curiosity_count = sum(
            1 for word in hook_text.split() if word in curiosity_words
        )
        features["transcript_hook_curiosity_words_count"] = curiosity_count

        # 2. Hook Action Words
        hook_action_words = [
            "get",
            "buy",
            "try",
            "start",
            "join",
            "learn",
            "discover",
            "find",
            "save",
            "win",
            "earn",
            "build",
            "create",
            "make",
            "take",
        ]
        hook_action_count = sum(
            1 for word in hook_text.split() if word in hook_action_words
        )
        features["transcript_hook_action_words_count"] = hook_action_count

        # 3. Hook Personal Pronouns
        personal_pronouns = ["you", "your", "yours", "we", "our", "us"]
        personal_count = sum(
            1 for word in hook_text.split() if word in personal_pronouns
        )
        features["transcript_hook_personal_pronouns_count"] = personal_count

        # 4. Hook Sentiment
        try:
            hook_blob = TextBlob(hook_text)
            features["transcript_hook_sentiment_polarity"] = float(
                hook_blob.sentiment.polarity
            )
        except:
            features["transcript_hook_sentiment_polarity"] = 0.0

        # Core Marketing Features

        # 5. Call to Action Count
        cta_phrases = [
            "buy now",
            "order now",
            "get now",
            "try now",
            "download now",
            "subscribe",
            "click here",
            "visit",
            "call now",
            "text now",
            "sign up",
            "register",
            "join now",
            "start now",
        ]
        cta_count = sum(1 for phrase in cta_phrases if phrase in text)
        features["transcript_call_to_action_count"] = cta_count

        # 6. Action Words Count
        action_words = [
            "get",
            "buy",
            "try",
            "start",
            "join",
            "learn",
            "discover",
            "find",
            "save",
            "win",
            "earn",
            "build",
            "create",
            "make",
            "take",
            "order",
            "download",
            "subscribe",
            "click",
            "visit",
            "call",
            "text",
            "sign",
            "register",
        ]
        action_count = sum(1 for word in words if word in action_words)
        features["transcript_action_words_count"] = action_count

        # 7. Overall Sentiment
        try:
            blob = TextBlob(transcript)
            features["transcript_sentiment_polarity"] = float(blob.sentiment.polarity)
        except:
            features["transcript_sentiment_polarity"] = 0.0

        # Engagement Features

        # 8. Exclamation Count
        features["transcript_exclamation_count"] = text.count("!")

        # 9. Question Count
        features["transcript_question_count"] = text.count("?")

        # 10. Word Count
        features["transcript_word_count"] = len(words)

        return features

    def _get_empty_transcript_features(self) -> dict[str, float]:
        """Return empty transcript features dict."""
        return {
            "transcript_hook_curiosity_words_count": 0.0,
            "transcript_hook_action_words_count": 0.0,
            "transcript_hook_personal_pronouns_count": 0.0,
            "transcript_hook_sentiment_polarity": 0.0,
            "transcript_call_to_action_count": 0.0,
            "transcript_action_words_count": 0.0,
            "transcript_sentiment_polarity": 0.0,
            "transcript_exclamation_count": 0.0,
            "transcript_question_count": 0.0,
            "transcript_word_count": 0.0,
        }

    def prepare_training_data(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Prepare training data from YouTube records (same 313 videos as metadata model)."""
        self.logger.info(
            "Preparing content-based training data from 313-video dataset..."
        )

        training_data = []
        processed_count = 0

        for record in records:
            try:
                video_id = record.get("video_id", "")
                if not video_id:
                    continue

                # Check if video file exists (same filter as simple trainer)
                video_path = self.video_cache_dir / f"{video_id}.mp4"
                if not video_path.exists():
                    continue

                self.stats["videos_with_files"] += 1
                processed_count += 1

                self.logger.info(f"Processing video {processed_count}: {video_id}")

                # Extract audio from video
                audio_path = self.extract_audio_from_video(str(video_path))
                self.stats["audio_extracted"] += 1

                # Extract audio features
                audio_features = self.extract_audio_features(audio_path)

                # Extract transcript
                transcript = self.extract_transcript(audio_path)
                if transcript and len(transcript.strip()) >= 10:
                    self.stats["transcripts_extracted"] += 1

                # Extract transcript features
                transcript_features = self.extract_transcript_features(transcript)

                # Clean up temp audio file
                try:
                    os.remove(audio_path)
                except:
                    pass

                # Combine features
                feature_row = {}

                # Add audio features
                for key, value in audio_features.items():
                    feature_row[key] = value

                # Add transcript features
                for key, value in transcript_features.items():
                    feature_row[key] = value

                # Add label (same logic as simple trainer)
                feature_row["performance_label"] = self.create_performance_label(record)
                feature_row["video_id"] = video_id
                feature_row["brand_name"] = record.get("brand_name", "unknown")

                training_data.append(feature_row)

            except Exception as e:
                self.logger.error(f"Error processing video {video_id}: {e}")
                continue

        if not training_data:
            raise ValueError("No training data could be extracted!")

        df = pd.DataFrame(training_data)

        # Update statistics
        self.stats["training_samples"] = len(df)
        self.stats["high_performance"] = len(df[df["performance_label"] == "high"])
        self.stats["low_performance"] = len(df[df["performance_label"] == "low"])

        self.logger.info(
            f"Created content-based training dataset with {len(df)} samples"
        )
        self.logger.info(f"High performance: {self.stats['high_performance']}")
        self.logger.info(f"Low performance: {self.stats['low_performance']}")

        return df

    def train_models(self, df: pd.DataFrame) -> dict[str, Any]:
        """Train multiple binary classifiers and return results."""
        self.logger.info(
            "Training content-based binary classifiers on 313-video dataset..."
        )

        # Prepare features and labels
        feature_columns = [
            col for col in df.columns if col.startswith(("audio_", "transcript_"))
        ]
        X = df[feature_columns]
        y = (df["performance_label"] == "high").astype(int)

        self.logger.info(
            f"Training with {len(feature_columns)} content-based features:"
        )
        self.logger.info(
            f"  Audio features: {len([c for c in feature_columns if c.startswith('audio_')])}"
        )
        self.logger.info(
            f"  Transcript features: {len([c for c in feature_columns if c.startswith('transcript_')])}"
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train models
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42),
        }

        results = {}

        for name, model in models.items():
            self.logger.info(f"Training {name}...")

            # Train model
            model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            # Cross-validation
            cv_folds = min(5, len(X_train))
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)

            results[name] = {
                "model": model,
                "accuracy": accuracy,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "classification_report": classification_report(
                    y_test, y_pred, output_dict=True
                ),
            }

            self.logger.info(
                f"{name} - Accuracy: {accuracy:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}"
            )

        # Find best model by CV performance
        best_model_name = max(results.keys(), key=lambda k: results[k]["cv_mean"])
        best_model = results[best_model_name]["model"]

        # Save best model and scaler
        joblib.dump(best_model, self.models_dir / "content_based_313_model.pkl")
        joblib.dump(scaler, self.models_dir / "content_based_313_scaler.pkl")
        joblib.dump(feature_columns, self.models_dir / "content_based_313_features.pkl")

        self.logger.info(f"Best model: {best_model_name}")
        self.logger.info("Models saved to ../models/")

        return {
            "results": results,
            "best_model": best_model_name,
            "feature_columns": feature_columns,
            "scaler": scaler,
        }

    def save_results(self, training_results: dict[str, Any], df: pd.DataFrame):
        """Save training results and statistics."""
        self.logger.info("Saving results...")

        # Convert results for JSON serialization
        json_results = {
            "dataset": "313_video_content_based",
            "statistics": self.stats,
            "best_model": training_results["best_model"],
            "feature_count": len(training_results["feature_columns"]),
            "features": training_results["feature_columns"],
            "model_performance": {},
        }

        for name, result in training_results["results"].items():
            json_results["model_performance"][name] = {
                "accuracy": float(result["accuracy"]),
                "cv_mean": float(result["cv_mean"]),
                "cv_std": float(result["cv_std"]),
            }

        # Save results
        with open(
            self.results_dir / "content_based_313_training_results.json", "w"
        ) as f:
            json.dump(json_results, f, indent=2)

        # Save training data for analysis
        df.to_csv(self.results_dir / "content_based_313_training_data.csv", index=False)

        self.logger.info("Results saved to ../results/")

    def run_training(self) -> dict[str, Any]:
        """Run the complete content-based training pipeline on 313 videos."""
        try:
            # Load data
            records = self.load_youtube_data()
            if not records:
                raise ValueError("No YouTube data loaded")

            # Create training dataset (same filter as simple trainer)
            df = self.prepare_training_data(records)

            # Train models
            training_results = self.train_models(df)

            # Save results
            self.save_results(training_results, df)

            return training_results

        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise
        finally:
            # Clean up temp directory
            try:
                import shutil

                shutil.rmtree(self.temp_dir)
            except:
                pass


def main():
    """Main training script."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🎬 Starting Content-Based Training on 313-Video Dataset")
    logger.info("=" * 70)
    logger.info("Training on same videos that achieved 96.8% metadata accuracy")

    try:
        trainer = ContentBased313Trainer()

        # Run training on the same 313 videos
        results = trainer.run_training()

        logger.info("🎉 Training completed successfully!")
        logger.info(f"Best model: {results['best_model']}")
        logger.info("=" * 70)
        logger.info("📊 COMPARISON OPPORTUNITY:")
        logger.info("  - Metadata Model (313 videos): 96.8% accuracy")
        logger.info(
            f"  - Content Model (313 videos): {results['results'][results['best_model']]['accuracy']:.1%} accuracy"
        )

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
