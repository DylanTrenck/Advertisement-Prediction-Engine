"""
Audio feature extraction using librosa and other audio processing libraries.
"""

import logging
import warnings

import librosa
import numpy as np

from .types import AudioFeatures, ProcessingConfig

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


class AudioFeatureExtractor:
    """
    Extracts the 10 most critical audio features for advertisement performance prediction.

    Streamlined features focus on:
    - Energy/attention-grabbing qualities (RMS, dynamic range)
    - Voice characteristics (pitch, speech rate)
    - Audio quality perception (spectral centroid, MFCCs 1-2)
    - Temporal flow and engagement (pauses, onset rate, zero crossing)

    Examples:
        >>> extractor = AudioFeatureExtractor()
        >>> features = extractor.extract_features("audio.wav")
        >>> assert isinstance(features, dict)
        >>> assert len(features) == 10  # Exactly 10 features
        >>> assert "rms_mean" in features
    """

    def __init__(self, config: ProcessingConfig | None = None):
        """
        Initialize the audio feature extractor.

        Args:
            config: Processing configuration with sample rate, window sizes, etc.

        Examples:
            >>> extractor = AudioFeatureExtractor()
            >>> assert extractor.sample_rate == 22050  # Default
        """
        self.logger = logging.getLogger(__name__)

        # Set configuration defaults
        self.sample_rate = config.get("sample_rate", 22050) if config else 22050
        self.frame_length = config.get("frame_length", 2048) if config else 2048
        self.hop_length = config.get("hop_length", 512) if config else 512
        self.n_mfcc = 2  # Only extract first 2 MFCCs (most important)
        self.n_fft = config.get("n_fft", 2048) if config else 2048

    def extract_features(self, audio_path: str) -> AudioFeatures:
        """
        Extract the 10 most important audio features from an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dictionary containing exactly 10 streamlined audio features

        Examples:
            >>> extractor = AudioFeatureExtractor()
            >>> features = extractor.extract_features("test_audio.wav")
            >>> assert len(features) == 10
            >>> assert "rms_mean" in features
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

            if len(y) == 0:
                self.logger.error(f"Empty audio file: {audio_path}")
                return self._get_empty_features()

            self.logger.info(
                f"Extracting streamlined features from {audio_path} ({len(y) / sr:.2f}s)"
            )

            # Initialize features dictionary
            features = AudioFeatures()

            # Extract only the 10 most critical features
            features.update(self._extract_core_energy_features(y))
            features.update(self._extract_core_voice_features(y, sr))
            features.update(self._extract_core_spectral_features(y, sr))
            features.update(self._extract_core_mfcc_features(y, sr))
            features.update(self._extract_core_temporal_features(y, sr))

            self.logger.info(f"Extracted {len(features)} streamlined audio features")
            return features

        except Exception as e:
            self.logger.error(f"Error extracting features from {audio_path}: {e}")
            return self._get_empty_features()

    def _extract_core_energy_features(self, y: np.ndarray) -> dict[str, float]:
        """Extract the 2 most critical energy features for advertisement impact."""
        features = {}

        # RMS energy - overall loudness/energy level (critical for attention-grabbing)
        rms = librosa.feature.rms(
            y=y, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]
        features["rms_mean"] = float(np.mean(rms))

        # Dynamic range - energy variation indicating excitement/drama
        features["dynamic_range"] = float(np.max(rms) - np.min(rms))

        return features

    def _extract_core_spectral_features(
        self, y: np.ndarray, sr: int
    ) -> dict[str, float]:
        """Extract the 1 most critical spectral feature for audio quality perception."""
        features = {}

        # Spectral centroid - brightness/perceived audio quality (critical for professional sound)
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=self.hop_length
        )[0]
        features["spectral_centroid_mean"] = float(np.mean(centroid))

        return features

    def _extract_core_mfcc_features(self, y: np.ndarray, sr: int) -> dict[str, float]:
        """Extract the 2 most important MFCC features for speech characteristics."""
        features = {}

        # Compute only first 2 MFCCs (most important for speech/voice characteristics)
        mfccs = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.n_mfcc,  # = 2
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Extract mean for first 2 MFCC coefficients (most predictive)
        features["mfcc_1_mean"] = float(
            np.mean(mfccs[0])
        )  # Primary speech characteristic
        features["mfcc_2_mean"] = float(
            np.mean(mfccs[1])
        )  # Secondary speech characteristic

        return features

    def _extract_core_voice_features(self, y: np.ndarray, sr: int) -> dict[str, float]:
        """Extract the 2 most critical voice characteristics for trust and comprehension."""
        features = {}

        try:
            # Pitch - voice characteristics affecting trust and authority
            pitches, magnitudes = librosa.piptrack(
                y=y, sr=sr, hop_length=self.hop_length, threshold=0.1
            )

            # Extract pitch values where magnitude is significant
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:  # Valid pitch
                    pitch_values.append(pitch)

            if pitch_values:
                features["pitch_mean"] = float(np.mean(pitch_values))
            else:
                features["pitch_mean"] = 0.0

            # Speech rate - speaking pace affecting comprehension and engagement
            onset_frames = librosa.onset.onset_detect(
                y=y, sr=sr, hop_length=self.hop_length
            )
            speech_events_per_second = len(onset_frames) / (len(y) / sr)
            features["speech_rate"] = float(speech_events_per_second)

        except Exception as e:
            self.logger.warning(f"Error extracting voice features: {e}")
            features.update({"pitch_mean": 0.0, "speech_rate": 0.0})

        return features

    def _extract_core_temporal_features(
        self, y: np.ndarray, sr: int
    ) -> dict[str, float]:
        """Extract the 3 most critical temporal features for engagement and flow."""
        features = {}

        try:
            # Pause duration - speaking rhythm affecting perceived flow
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
            rms_threshold = np.mean(rms) * 0.1  # 10% of mean RMS
            pause_frames = rms < rms_threshold
            pause_segments = self._find_segments(pause_frames)

            if pause_segments:
                pause_durations = [
                    (end - start) * self.hop_length / sr
                    for start, end in pause_segments
                ]
                features["pause_duration_mean"] = float(np.mean(pause_durations))
            else:
                features["pause_duration_mean"] = 0.0

            # Onset rate - event density indicating activity level and engagement
            onset_frames = librosa.onset.onset_detect(
                y=y, sr=sr, hop_length=self.hop_length, units="time"
            )
            duration = len(y) / sr
            features["onset_rate"] = float(len(onset_frames) / duration)

            # Zero crossing rate - speech vs music distinction (important for voice clarity)
            zcr = librosa.feature.zero_crossing_rate(
                y, frame_length=self.frame_length, hop_length=self.hop_length
            )[0]
            features["zero_crossing_rate"] = float(np.mean(zcr))

        except Exception as e:
            self.logger.warning(f"Error extracting temporal features: {e}")
            features.update(
                {
                    "pause_duration_mean": 0.0,
                    "onset_rate": 0.0,
                    "zero_crossing_rate": 0.0,
                }
            )

        return features

    def _find_segments(self, boolean_array: np.ndarray) -> list[tuple[int, int]]:
        """Find continuous segments where boolean_array is True."""
        segments = []
        start = None

        for i, value in enumerate(boolean_array):
            if value and start is None:
                start = i
            elif not value and start is not None:
                segments.append((start, i))
                start = None

        # Handle case where segment extends to end
        if start is not None:
            segments.append((start, len(boolean_array)))

        return segments

    def _get_empty_features(self) -> AudioFeatures:
        """Return empty/default features for error cases - exactly 10 features."""
        return AudioFeatures(
            # Energy features (2)
            rms_mean=0.0,
            dynamic_range=0.0,
            # Voice features (2)
            pitch_mean=0.0,
            speech_rate=0.0,
            # Spectral features (1)
            spectral_centroid_mean=0.0,
            # MFCC features (2)
            mfcc_1_mean=0.0,
            mfcc_2_mean=0.0,
            # Temporal features (3)
            pause_duration_mean=0.0,
            onset_rate=0.0,
            zero_crossing_rate=0.0,
        )

    def extract_features_batch(self, audio_paths: list[str]) -> list[AudioFeatures]:
        """
        Extract features from multiple audio files.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            List of feature dictionaries

        Examples:
            >>> extractor = AudioFeatureExtractor()
            >>> paths = ["audio1.wav", "audio2.wav"]
            >>> features_list = extractor.extract_features_batch(paths)
            >>> assert len(features_list) == len(paths)
        """
        results = []

        for audio_path in audio_paths:
            self.logger.info(f"Processing {audio_path}")
            features = self.extract_features(audio_path)
            results.append(features)

        return results
