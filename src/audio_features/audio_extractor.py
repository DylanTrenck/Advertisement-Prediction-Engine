"""
Audio extraction from video files using FFmpeg.
"""

import logging
import tempfile
from pathlib import Path

import ffmpeg
from pydub import AudioSegment

from .types import AudioProcessingResult, ProcessingConfig


class AudioExtractor:
    """
    Extracts audio from MP4 and other video files using FFmpeg.

    Examples:
        >>> extractor = AudioExtractor()
        >>> result = extractor.extract_audio("video.mp4", "output.wav")
        >>> assert result.success == True
        >>> assert Path(result.audio_file_path).exists()
    """

    def __init__(self, temp_dir: str | None = None):
        """
        Initialize the audio extractor.

        Args:
            temp_dir: Directory for temporary files. If None, uses system temp.

        Examples:
            >>> extractor = AudioExtractor()
            >>> assert extractor.temp_dir is not None
        """
        self.logger = logging.getLogger(__name__)
        self.temp_dir = temp_dir or tempfile.gettempdir()

        # Ensure temp directory exists
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

    def extract_audio_from_mp4(
        self,
        video_path: str,
        output_path: str | None = None,
        config: ProcessingConfig | None = None,
    ) -> AudioProcessingResult:
        """
        Extract audio from MP4 video file.

        Args:
            video_path: Path to the input video file
            output_path: Path for output audio file. If None, creates temp file.
            config: Processing configuration

        Returns:
            Result containing success status and extracted audio info

        Examples:
            >>> extractor = AudioExtractor()
            >>> result = extractor.extract_audio_from_mp4("test.mp4")
            >>> assert result.success in [True, False]  # Depends on file existence
            >>> if result.success:
            ...     assert result.duration_seconds > 0
        """
        import time

        start_time = time.time()

        try:
            # Validate input file
            if not Path(video_path).exists():
                return AudioProcessingResult(
                    success=False,
                    audio_file_path="",
                    sample_rate=0,
                    duration_seconds=0.0,
                    channels=0,
                    features_extracted=False,
                    transcript_extracted=False,
                    error_message=f"Video file not found: {video_path}",
                    processing_time_seconds=time.time() - start_time,
                )

            # Set default config values
            sample_rate = config.get("sample_rate", 22050) if config else 22050

            # Create output path if not provided
            if output_path is None:
                video_name = Path(video_path).stem
                output_path = str(Path(self.temp_dir) / f"{video_name}_audio.wav")

            # Extract audio using FFmpeg
            self.logger.info(f"Extracting audio from {video_path}")

            stream = ffmpeg.input(video_path)
            audio = stream.audio

            # Configure audio output
            output_stream = ffmpeg.output(
                audio,
                output_path,
                acodec="pcm_s16le",  # 16-bit PCM
                ar=sample_rate,  # Sample rate
                ac=1,  # Mono channel
            )

            # Run FFmpeg extraction
            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)

            # Verify output file was created
            if not Path(output_path).exists():
                return AudioProcessingResult(
                    success=False,
                    audio_file_path="",
                    sample_rate=0,
                    duration_seconds=0.0,
                    channels=0,
                    features_extracted=False,
                    transcript_extracted=False,
                    error_message="Failed to create audio output file",
                    processing_time_seconds=time.time() - start_time,
                )

            # Get audio metadata using pydub
            audio_segment = AudioSegment.from_wav(output_path)
            duration_seconds = len(audio_segment) / 1000.0  # Convert ms to seconds
            channels = audio_segment.channels
            actual_sample_rate = audio_segment.frame_rate

            self.logger.info(
                f"Successfully extracted audio: {duration_seconds:.2f}s, "
                f"{actual_sample_rate}Hz, {channels} channel(s)"
            )

            return AudioProcessingResult(
                success=True,
                audio_file_path=output_path,
                sample_rate=actual_sample_rate,
                duration_seconds=duration_seconds,
                channels=channels,
                features_extracted=False,  # Will be set by downstream components
                transcript_extracted=False,  # Will be set by downstream components
                error_message=None,
                processing_time_seconds=time.time() - start_time,
            )

        except ffmpeg.Error as e:
            error_msg = f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}"
            self.logger.error(error_msg)

            return AudioProcessingResult(
                success=False,
                audio_file_path="",
                sample_rate=0,
                duration_seconds=0.0,
                channels=0,
                features_extracted=False,
                transcript_extracted=False,
                error_message=error_msg,
                processing_time_seconds=time.time() - start_time,
            )

        except Exception as e:
            error_msg = f"Unexpected error during audio extraction: {str(e)}"
            self.logger.error(error_msg)

            return AudioProcessingResult(
                success=False,
                audio_file_path="",
                sample_rate=0,
                duration_seconds=0.0,
                channels=0,
                features_extracted=False,
                transcript_extracted=False,
                error_message=error_msg,
                processing_time_seconds=time.time() - start_time,
            )

    def get_video_info(self, video_path: str) -> dict[str, any]:
        """
        Get metadata information about a video file.

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary containing video metadata

        Examples:
            >>> extractor = AudioExtractor()
            >>> info = extractor.get_video_info("test.mp4")
            >>> assert isinstance(info, dict)
        """
        try:
            probe = ffmpeg.probe(video_path)

            # Find audio and video streams
            audio_stream = None
            video_stream = None

            for stream in probe["streams"]:
                if stream["codec_type"] == "audio" and audio_stream is None:
                    audio_stream = stream
                elif stream["codec_type"] == "video" and video_stream is None:
                    video_stream = stream

            # Extract relevant information
            info = {
                "file_path": video_path,
                "file_size_mb": Path(video_path).stat().st_size / (1024 * 1024),
                "duration_seconds": float(probe["format"].get("duration", 0)),
                "video_codec": video_stream.get("codec_name", "unknown")
                if video_stream
                else "none",
                "audio_codec": audio_stream.get("codec_name", "unknown")
                if audio_stream
                else "none",
                "sample_rate": int(audio_stream.get("sample_rate", 0))
                if audio_stream
                else 0,
                "channels": int(audio_stream.get("channels", 0)) if audio_stream else 0,
                "bitrate": int(probe["format"].get("bit_rate", 0)),
            }

            return info

        except Exception as e:
            self.logger.error(f"Error getting video info for {video_path}: {e}")
            return {"file_path": video_path, "error": str(e)}

    def cleanup_temp_files(self, keep_recent_hours: int = 24) -> int:
        """
        Clean up old temporary audio files.

        Args:
            keep_recent_hours: Keep files created within this many hours

        Returns:
            Number of files deleted

        Examples:
            >>> extractor = AudioExtractor()
            >>> deleted_count = extractor.cleanup_temp_files(24)
            >>> assert deleted_count >= 0
        """
        import time

        deleted_count = 0
        current_time = time.time()
        keep_threshold = keep_recent_hours * 3600  # Convert to seconds

        try:
            temp_path = Path(self.temp_dir)

            for file_path in temp_path.glob("*_audio.wav"):
                file_age = current_time - file_path.stat().st_mtime

                if file_age > keep_threshold:
                    file_path.unlink()
                    deleted_count += 1
                    self.logger.debug(f"Deleted old temp file: {file_path}")

            self.logger.info(f"Cleaned up {deleted_count} temporary audio files")

        except Exception as e:
            self.logger.error(f"Error during temp file cleanup: {e}")

        return deleted_count
