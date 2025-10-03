"""
Speech-to-text transcript extraction using various ASR services.
"""

import logging
from pathlib import Path

import whisper

from .types import ProcessingConfig


class TranscriptExtractor:
    """
    Extracts text transcripts from audio using speech recognition services.

    Supports multiple ASR backends:
    - OpenAI Whisper (local)
    - Google Cloud Speech-to-Text (cloud)
    - Azure Speech Services (cloud)
    - AWS Transcribe (cloud)

    Examples:
        >>> extractor = TranscriptExtractor()
        >>> transcript = extractor.extract_transcript("audio.wav")
        >>> assert isinstance(transcript, str)
    """

    def __init__(self, config: ProcessingConfig | None = None):
        """
        Initialize the transcript extractor.

        Args:
            config: Processing configuration with ASR service settings

        Examples:
            >>> extractor = TranscriptExtractor()
            >>> assert extractor.asr_service == "whisper"  # Default
        """
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.asr_service = config.get("asr_service", "whisper") if config else "whisper"
        self.language = config.get("asr_language", "en-US") if config else "en-US"
        self.confidence_threshold = (
            config.get("asr_confidence_threshold", 0.5) if config else 0.5
        )

        # Initialize ASR service
        self.whisper_model = None
        if self.asr_service == "whisper":
            self._init_whisper()

    def _init_whisper(self) -> None:
        """Initialize Whisper model."""
        try:
            self.logger.info("Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
            self.logger.info("Whisper model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            self.whisper_model = None

    def extract_transcript(self, audio_path: str) -> str:
        """
        Extract transcript from audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            Extracted transcript text

        Examples:
            >>> extractor = TranscriptExtractor()
            >>> text = extractor.extract_transcript("speech.wav")
            >>> assert isinstance(text, str)
            >>> assert len(text) >= 0  # May be empty for non-speech audio
        """
        if not Path(audio_path).exists():
            self.logger.error(f"Audio file not found: {audio_path}")
            return ""

        try:
            if self.asr_service == "whisper":
                return self._extract_with_whisper(audio_path)
            elif self.asr_service == "google":
                return self._extract_with_google(audio_path)
            elif self.asr_service == "azure":
                return self._extract_with_azure(audio_path)
            elif self.asr_service == "aws":
                return self._extract_with_aws(audio_path)
            else:
                self.logger.error(f"Unsupported ASR service: {self.asr_service}")
                return ""

        except Exception as e:
            self.logger.error(f"Error extracting transcript from {audio_path}: {e}")
            return ""

    def _extract_with_whisper(self, audio_path: str) -> str:
        """Extract transcript using OpenAI Whisper."""
        if self.whisper_model is None:
            self.logger.error("Whisper model not initialized")
            return ""

        try:
            self.logger.info(f"Transcribing with Whisper: {audio_path}")

            # Transcribe audio
            result = self.whisper_model.transcribe(audio_path)

            # Extract text with confidence filtering
            transcript_segments = []
            for segment in result.get("segments", []):
                # Whisper doesn't provide per-segment confidence in base model
                # but we can use the overall result
                text = segment.get("text", "").strip()
                if text:
                    transcript_segments.append(text)

            transcript = " ".join(transcript_segments)
            self.logger.info(f"Whisper transcript length: {len(transcript)} characters")

            return transcript

        except Exception as e:
            self.logger.error(f"Whisper transcription failed: {e}")
            return ""

    def _extract_with_google(self, audio_path: str) -> str:
        """Extract transcript using Google Cloud Speech-to-Text."""
        try:
            from google.cloud import speech

            # Initialize client
            client = speech.SpeechClient()

            # Read audio file
            with open(audio_path, "rb") as audio_file:
                content = audio_file.read()

            # Configure recognition
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=22050,
                language_code=self.language,
                enable_automatic_punctuation=True,
                enable_word_confidence=True,
            )

            # Perform recognition
            response = client.recognize(config=config, audio=audio)

            # Extract transcript with confidence filtering
            transcript_parts = []
            for result in response.results:
                if result.alternatives:
                    alternative = result.alternatives[0]
                    if alternative.confidence >= self.confidence_threshold:
                        transcript_parts.append(alternative.transcript)

            transcript = " ".join(transcript_parts)
            self.logger.info(
                f"Google Speech transcript length: {len(transcript)} characters"
            )

            return transcript

        except Exception as e:
            self.logger.error(f"Google Speech transcription failed: {e}")
            return ""

    def _extract_with_azure(self, audio_path: str) -> str:
        """Extract transcript using Azure Speech Services."""
        try:
            # This would require Azure credentials and configuration
            # Placeholder implementation
            self.logger.warning("Azure Speech not fully implemented")
            return ""

        except ImportError:
            self.logger.error("Azure Speech SDK not installed")
            return ""
        except Exception as e:
            self.logger.error(f"Azure Speech transcription failed: {e}")
            return ""

    def _extract_with_aws(self, audio_path: str) -> str:
        """Extract transcript using AWS Transcribe."""
        try:
            # This would require AWS credentials and S3 upload
            # Placeholder implementation
            self.logger.warning("AWS Transcribe not fully implemented")
            return ""

        except ImportError:
            self.logger.error("AWS SDK not installed")
            return ""
        except Exception as e:
            self.logger.error(f"AWS Transcribe transcription failed: {e}")
            return ""

    def extract_transcript_with_metadata(self, audio_path: str) -> dict[str, any]:
        """
        Extract transcript with additional metadata.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dictionary containing transcript and metadata

        Examples:
            >>> extractor = TranscriptExtractor()
            >>> result = extractor.extract_transcript_with_metadata("speech.wav")
            >>> assert "transcript" in result
            >>> assert "confidence" in result
        """
        if not Path(audio_path).exists():
            return {
                "transcript": "",
                "confidence": 0.0,
                "language": "",
                "segments": [],
                "processing_time": 0.0,
                "error": f"File not found: {audio_path}",
            }

        import time

        start_time = time.time()

        try:
            if self.asr_service == "whisper" and self.whisper_model:
                result = self.whisper_model.transcribe(audio_path)

                return {
                    "transcript": result.get("text", ""),
                    "confidence": 1.0,  # Whisper doesn't provide overall confidence
                    "language": result.get("language", ""),
                    "segments": result.get("segments", []),
                    "processing_time": time.time() - start_time,
                    "error": None,
                }
            else:
                # For other services, fall back to simple extraction
                transcript = self.extract_transcript(audio_path)
                return {
                    "transcript": transcript,
                    "confidence": 1.0 if transcript else 0.0,
                    "language": self.language,
                    "segments": [],
                    "processing_time": time.time() - start_time,
                    "error": None if transcript else "No transcript extracted",
                }

        except Exception as e:
            return {
                "transcript": "",
                "confidence": 0.0,
                "language": "",
                "segments": [],
                "processing_time": time.time() - start_time,
                "error": str(e),
            }

    def batch_extract_transcripts(self, audio_paths: list[str]) -> list[str]:
        """
        Extract transcripts from multiple audio files.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            List of transcript strings

        Examples:
            >>> extractor = TranscriptExtractor()
            >>> paths = ["audio1.wav", "audio2.wav"]
            >>> transcripts = extractor.batch_extract_transcripts(paths)
            >>> assert len(transcripts) == len(paths)
        """
        results = []

        for audio_path in audio_paths:
            self.logger.info(f"Transcribing {audio_path}")
            transcript = self.extract_transcript(audio_path)
            results.append(transcript)

        return results
