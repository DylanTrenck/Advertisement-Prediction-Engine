"""
Text feature extraction from transcripts for advertisement analysis.
"""

import logging
import re

import nltk
from textblob import TextBlob

from .types import ProcessingConfig, TranscriptFeatures

# Download required NLTK data
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
except:
    pass


class TranscriptFeatureExtractor:
    """
    Extracts the 10 most critical text features from advertisement transcripts.

    Streamlined features focus on:
    - Hook analysis (4 features) - Critical first 3-5 seconds for retention
    - Core marketing elements (3 features) - Direct conversion drivers
    - Engagement indicators (3 features) - Retention and interaction factors

    Examples:
        >>> extractor = TranscriptFeatureExtractor()
        >>> features = extractor.extract_features("Buy now! Amazing deal!")
        >>> assert len(features) == 10  # Exactly 10 features
        >>> assert "call_to_action_count" in features
    """

    def __init__(self, config: ProcessingConfig | None = None):
        """Initialize the transcript feature extractor."""
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.remove_stopwords = (
            config.get("remove_stopwords", False) if config else False
        )
        self.stem_words = config.get("stem_words", False) if config else False
        self.max_text_length = config.get("max_text_length", 10000) if config else 10000

        # Load stopwords and stemmer
        try:
            from nltk.corpus import stopwords
            from nltk.stem import PorterStemmer

            self.stopwords = set(stopwords.words("english"))
            self.stemmer = PorterStemmer()
        except:
            self.stopwords = set()
            self.stemmer = None

        # Predefined word lists for marketing analysis
        self.action_words = {
            "buy",
            "purchase",
            "order",
            "shop",
            "get",
            "try",
            "visit",
            "call",
            "click",
            "sign up",
            "subscribe",
            "download",
            "learn",
            "discover",
            "explore",
            "save",
            "join",
            "start",
            "begin",
            "choose",
            "select",
            "pick",
            "grab",
            "take",
        }

        self.product_words = {
            "product",
            "service",
            "solution",
            "offer",
            "deal",
            "discount",
            "sale",
            "price",
            "cost",
            "value",
            "quality",
            "feature",
            "benefit",
            "advantage",
        }

        self.cta_patterns = [
            r"\b(call now|buy now|order now|click here|sign up|learn more)\b",
            r"\b(limited time|act fast|don\'t wait|hurry|today only)\b",
            r"\b(free shipping|money back|guarantee|risk free)\b",
        ]

        # Hook-specific word lists
        self.curiosity_words = {
            "secret",
            "mystery",
            "hidden",
            "revealed",
            "discover",
            "uncover",
            "expose",
            "truth",
            "shocking",
            "surprising",
            "amazing",
            "incredible",
            "unbelievable",
            "weird",
            "strange",
            "bizarre",
            "mysterious",
            "forbidden",
            "exclusive",
        }

        self.power_words = {
            "instant",
            "immediately",
            "guaranteed",
            "proven",
            "revolutionary",
            "breakthrough",
            "ultimate",
            "perfect",
            "exclusive",
            "limited",
            "premium",
            "professional",
            "advanced",
            "superior",
            "cutting-edge",
            "state-of-the-art",
            "world-class",
        }

        self.personal_pronouns = {
            "you",
            "your",
            "yours",
            "yourself",
            "we",
            "us",
            "our",
            "ours",
        }

    def extract_features(self, transcript: str) -> TranscriptFeatures:
        """
        Extract the 10 most critical text features from a transcript.

        Args:
            transcript: The transcript text to analyze

        Returns:
            Dictionary containing exactly 10 streamlined text features
        """
        if not transcript or not transcript.strip():
            return self._get_empty_features()

        # Truncate if too long
        if len(transcript) > self.max_text_length:
            transcript = transcript[: self.max_text_length]

        features = TranscriptFeatures()

        # Extract only the 10 most critical features
        features.update(self._extract_core_hook_features(transcript))  # 4 features
        features.update(self._extract_core_marketing_features(transcript))  # 3 features
        features.update(
            self._extract_core_engagement_features(transcript)
        )  # 3 features

        self.logger.info(f"Extracted {len(features)} streamlined transcript features")
        return features

    def _extract_core_hook_features(self, text: str) -> dict[str, any]:
        """
        Extract the 4 most critical hook features (first 3-5 seconds).
        These determine if viewers continue watching or scroll away.
        """
        features = {}

        # Extract the hook (first sentence or ~10-15 words, whichever is shorter)
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        first_sentence = sentences[0].strip() if sentences else text

        # Hook is either first sentence or first 15 words, whichever is shorter
        hook_words = first_sentence.split()
        if len(hook_words) > 15:
            hook_words = words[:15]

        hook_text = " ".join(hook_words)
        hook_text.lower()

        # 1. Curiosity/intrigue words (crucial for retention)
        curiosity_count = sum(
            1 for word in hook_words if word.lower() in self.curiosity_words
        )
        features["hook_curiosity_words_count"] = curiosity_count

        # 2. Action orientation in hook (immediate engagement)
        hook_action_count = sum(
            1 for word in hook_words if word.lower() in self.action_words
        )
        features["hook_action_words_count"] = hook_action_count

        # 3. Personal connection (you, your, we, our)
        personal_count = sum(
            1 for word in hook_words if word.lower() in self.personal_pronouns
        )
        features["hook_personal_pronouns_count"] = personal_count

        # 4. Hook sentiment (critical for first impression)
        try:
            from textblob import TextBlob

            hook_blob = TextBlob(hook_text)
            features["hook_sentiment_polarity"] = float(hook_blob.sentiment.polarity)
        except:
            features["hook_sentiment_polarity"] = 0.0

        return features

    def _extract_core_marketing_features(self, text: str) -> dict[str, float]:
        """
        Extract the 3 most critical marketing features for conversion.
        """
        features = {}

        text_lower = text.lower()
        words = text_lower.split()

        # 1. Call-to-action detection (direct conversion driver)
        cta_count = 0
        for pattern in self.cta_patterns:
            cta_count += len(re.findall(pattern, text_lower, re.IGNORECASE))
        features["call_to_action_count"] = cta_count

        # 2. Action words (total action-oriented language)
        action_count = sum(1 for word in words if word in self.action_words)
        features["action_words_count"] = action_count

        # 3. Overall sentiment polarity (emotional tone affecting perception)
        try:
            blob = TextBlob(text)
            features["sentiment_polarity"] = float(blob.sentiment.polarity)
        except:
            features["sentiment_polarity"] = 0.0

        return features

    def _extract_core_engagement_features(self, text: str) -> dict[str, float]:
        """
        Extract the 3 most critical engagement features for retention.
        """
        features = {}

        # 1. Exclamation count - energy and excitement level
        features["exclamation_count"] = text.count("!")

        # 2. Question count - audience engagement techniques
        features["question_count"] = text.count("?")

        # 3. Word count - content density and pacing
        words = text.split()
        features["word_count"] = len(words)

        return features

    def _get_empty_features(self) -> TranscriptFeatures:
        """Return empty features for error cases - exactly 10 features."""
        return TranscriptFeatures(
            # Hook Analysis (4 features)
            hook_curiosity_words_count=0,
            hook_action_words_count=0,
            hook_personal_pronouns_count=0,
            hook_sentiment_polarity=0.0,
            # Core Marketing Elements (3 features)
            call_to_action_count=0,
            action_words_count=0,
            sentiment_polarity=0.0,
            # Engagement Indicators (3 features)
            exclamation_count=0,
            question_count=0,
            word_count=0,
        )
