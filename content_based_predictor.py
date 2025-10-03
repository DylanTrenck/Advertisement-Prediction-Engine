#!/usr/bin/env python3
"""
Content-Based YouTube Performance Predictor

Predicts video performance using ONLY audio and transcript features
extracted from uploaded video content.
"""

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
import speech_recognition as sr
from pydub import AudioSegment
from textblob import TextBlob


class ContentBasedPredictor:
    """
    Predicts YouTube video performance using content-based features only.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.temp_dir = tempfile.mkdtemp()

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()

        # Download NLTK data
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
        except:
            pass

        self.load_model()

    def load_model(self):
        """Load the trained content-based model (313-video trained)."""
        try:
            model_path = self.models_dir / "content_based_313_model.pkl"
            scaler_path = self.models_dir / "content_based_313_scaler.pkl"
            features_path = self.models_dir / "content_based_313_features.pkl"

            if not all(p.exists() for p in [model_path, scaler_path, features_path]):
                self.logger.error("Content-based model files not found!")
                raise FileNotFoundError("Model files missing")

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_columns = joblib.load(features_path)

            self.logger.info(
                f"Loaded content-based model (313 videos) with {len(self.feature_columns)} features"
            )

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video file."""
        try:
            audio_output_path = os.path.join(self.temp_dir, "extracted_audio.wav")

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

    def predict_performance(self, video_path: str) -> dict[str, Any]:
        """
        Predict video performance from uploaded video file.

        Args:
            video_path: Path to uploaded video file

        Returns:
            Dictionary with prediction results
        """
        try:
            self.logger.info(f"Analyzing video: {video_path}")

            # Extract audio from video
            audio_path = self.extract_audio_from_video(video_path)

            # Extract audio features
            audio_features = self.extract_audio_features(audio_path)

            # Extract transcript
            transcript = self.extract_transcript(audio_path)

            # Extract transcript features
            transcript_features = self.extract_transcript_features(transcript)

            # Combine all features
            all_features = {**audio_features, **transcript_features}

            # Create feature vector
            feature_vector = []
            for col in self.feature_columns:
                feature_vector.append(all_features.get(col, 0.0))

            # Scale features
            X = self.scaler.transform([feature_vector])

            # Make prediction
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]

            # Clean up temp audio file
            try:
                os.remove(audio_path)
            except:
                pass

            # Format results (include fields expected by frontend under binary_classification)
            result = {
                "prediction": "high" if prediction == 1 else "low",
                "confidence": float(max(probabilities)),
                "probabilities": {
                    "low": float(probabilities[0]),
                    "high": float(probabilities[1]),
                },
                "binary_classification": {
                    "label": "high" if prediction == 1 else "low",
                    "prediction": "high" if prediction == 1 else "low",
                    "confidence": float(max(probabilities)),
                    "score_high": float(probabilities[1]),
                    "threshold": 0.5,
                    "probabilities": {
                        "low": float(probabilities[0]),
                        "high": float(probabilities[1]),
                    },
                },
                "audio_features": audio_features,
                "transcript_features": transcript_features,
                "transcript_text": transcript,
                "features_analyzed": len(self.feature_columns),
                "model_type": "Content-Based Binary Classifier",
            }

            self.logger.info(
                f"Prediction: {result['prediction']} ({result['confidence']:.1%} confidence)"
            )

            return result

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}", exc_info=True)
            raise

    def __del__(self):
        """Clean up temporary directory."""
        try:
            import shutil

            shutil.rmtree(self.temp_dir)
        except:
            pass


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


if __name__ == "__main__":
    """Test the predictor with a sample video."""
    setup_logging()

    if len(sys.argv) != 2:
        print("Usage: python content_based_predictor.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    try:
        predictor = ContentBasedPredictor()
        result = predictor.predict_performance(video_path)

        print("\n🎯 Content-Based Prediction Results:")
        print("=" * 40)
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(
            f"Probabilities: High={result['probabilities']['high']:.1%}, Low={result['probabilities']['low']:.1%}"
        )
        print(f"Features Analyzed: {result['features_analyzed']}")
        print(f"Transcript: {result['transcript_text'][:100]}...")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
