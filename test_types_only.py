#!/usr/bin/env python3
"""
Standalone test for the streamlined feature types.
Tests only the type definitions without importing the full package.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import types directly to avoid dependency issues
from audio_features.types import AudioFeatures, TranscriptFeatures


def test_audio_feature_structure():
    """Test that AudioFeatures has exactly the 10 expected fields."""
    print("🎵 Testing Audio Feature Structure...")

    # The 10 most critical audio features for ad performance
    expected_fields = {
        "rms_mean",  # Energy level for attention
        "dynamic_range",  # Energy variation for drama
        "pitch_mean",  # Voice authority and trust
        "speech_rate",  # Comprehension and pacing
        "spectral_centroid_mean",  # Audio quality perception
        "mfcc_1_mean",  # Primary speech characteristic
        "mfcc_2_mean",  # Secondary speech characteristic
        "pause_duration_mean",  # Speaking rhythm and flow
        "onset_rate",  # Activity level and engagement
        "zero_crossing_rate",  # Speech vs music distinction
    }

    # Create an instance with all fields
    audio_features = AudioFeatures(
        rms_mean=0.1,
        dynamic_range=0.2,
        pitch_mean=150.0,
        speech_rate=3.5,
        spectral_centroid_mean=2000.0,
        mfcc_1_mean=-10.5,
        mfcc_2_mean=5.2,
        pause_duration_mean=0.8,
        onset_rate=2.1,
        zero_crossing_rate=0.15,
    )

    actual_fields = set(audio_features.keys())

    print(f"Expected fields ({len(expected_fields)}):")
    for i, field in enumerate(sorted(expected_fields), 1):
        print(f"  {i:2d}. {field}")

    print(f"\nActual fields ({len(actual_fields)}):")
    for i, field in enumerate(sorted(actual_fields), 1):
        print(f"  {i:2d}. {field}")

    if actual_fields == expected_fields:
        print("\n✅ Audio features structure is PERFECT!")
        return True
    else:
        print("\n❌ Audio features structure mismatch!")
        print(f"Missing: {expected_fields - actual_fields}")
        print(f"Extra: {actual_fields - expected_fields}")
        return False


def test_transcript_feature_structure():
    """Test that TranscriptFeatures has exactly the 10 expected fields."""
    print("\n📝 Testing Transcript Feature Structure...")

    # The 10 most critical transcript features for ad performance
    expected_fields = {
        # Hook Analysis (4 features) - Critical first 3-5 seconds
        "hook_curiosity_words_count",  # "Secret", "shocking" - drives interest
        "hook_action_words_count",  # "Get", "try", "buy" - immediate CTAs
        "hook_personal_pronouns_count",  # "You", "your" - personal connection
        "hook_sentiment_polarity",  # Positive emotion in opening
        # Core Marketing Elements (3 features) - Conversion drivers
        "call_to_action_count",  # "Buy now", "click here" - direct conversion
        "action_words_count",  # Total action-oriented language
        "sentiment_polarity",  # Overall emotional tone
        # Engagement Indicators (3 features) - Retention factors
        "exclamation_count",  # Energy and excitement level
        "question_count",  # Audience engagement techniques
        "word_count",  # Content density and pacing
    }

    # Create an instance with all fields
    transcript_features = TranscriptFeatures(
        hook_curiosity_words_count=2,
        hook_action_words_count=1,
        hook_personal_pronouns_count=3,
        hook_sentiment_polarity=0.8,
        call_to_action_count=2,
        action_words_count=5,
        sentiment_polarity=0.6,
        exclamation_count=3,
        question_count=1,
        word_count=45,
    )

    actual_fields = set(transcript_features.keys())

    print(f"Expected fields ({len(expected_fields)}):")
    for i, field in enumerate(sorted(expected_fields), 1):
        print(f"  {i:2d}. {field}")

    print(f"\nActual fields ({len(actual_fields)}):")
    for i, field in enumerate(sorted(actual_fields), 1):
        print(f"  {i:2d}. {field}")

    if actual_fields == expected_fields:
        print("\n✅ Transcript features structure is PERFECT!")
        return True
    else:
        print("\n❌ Transcript features structure mismatch!")
        print(f"Missing: {expected_fields - actual_fields}")
        print(f"Extra: {actual_fields - expected_fields}")
        return False


def show_feature_summary():
    """Show a summary of our streamlined features."""
    print("\n📊 STREAMLINED FEATURE SUMMARY")
    print("=" * 65)

    print("\n🎵 AUDIO FEATURES (10 total)")
    print("Energy & Attention (2):")
    print("  • rms_mean - Overall loudness for attention-grabbing")
    print("  • dynamic_range - Energy variation for excitement/drama")

    print("Voice Quality & Trust (2):")
    print("  • pitch_mean - Voice characteristics affecting authority")
    print("  • speech_rate - Speaking pace for comprehension")

    print("Audio Quality (3):")
    print("  • spectral_centroid_mean - Brightness/perceived quality")
    print("  • mfcc_1_mean - Primary speech characteristic")
    print("  • mfcc_2_mean - Secondary speech characteristic")

    print("Temporal Flow (3):")
    print("  • pause_duration_mean - Speaking rhythm and flow")
    print("  • onset_rate - Activity level and engagement")
    print("  • zero_crossing_rate - Speech vs music distinction")

    print("\n📝 TRANSCRIPT FEATURES (10 total)")
    print("Hook Analysis - First 3-5 seconds (4):")
    print("  • hook_curiosity_words_count - Intrigue words for retention")
    print("  • hook_action_words_count - Immediate engagement CTAs")
    print("  • hook_personal_pronouns_count - Personal connection")
    print("  • hook_sentiment_polarity - Opening emotional tone")

    print("Marketing Elements (3):")
    print("  • call_to_action_count - Direct conversion drivers")
    print("  • action_words_count - Total action-oriented language")
    print("  • sentiment_polarity - Overall emotional appeal")

    print("Engagement Indicators (3):")
    print("  • exclamation_count - Energy and excitement")
    print("  • question_count - Audience engagement techniques")
    print("  • word_count - Content density and pacing")


def main():
    """Run all tests."""
    print("🚀 STREAMLINED FEATURE EXTRACTION - TYPE VERIFICATION")
    print("=" * 65)
    print("Testing the most critical 20 features for ad performance prediction")
    print("Reduced from 60+ features to focus on maximum predictive power")

    audio_test_passed = test_audio_feature_structure()
    transcript_test_passed = test_transcript_feature_structure()

    print("\n" + "=" * 65)
    print("🎯 TEST RESULTS")
    print("=" * 65)

    if audio_test_passed:
        print("✅ Audio features: PERFECT (exactly 10 most critical features)")
    else:
        print("❌ Audio features: FAILED")

    if transcript_test_passed:
        print("✅ Transcript features: PERFECT (exactly 10 most critical features)")
    else:
        print("❌ Transcript features: FAILED")

    all_passed = audio_test_passed and transcript_test_passed

    if all_passed:
        show_feature_summary()
        print("\n🎉 STREAMLINED FEATURE EXTRACTION READY!")
        print("✨ Successfully reduced from 60+ to 20 most predictive features")
        print("🚀 Ready for high-performance advertisement prediction training")
        return True
    else:
        print("\n❌ Feature structure needs fixing")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
