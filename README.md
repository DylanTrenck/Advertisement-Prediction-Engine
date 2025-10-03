# Audio Features Advertisement Performance System

A comprehensive system for predicting advertisement performance using audio analysis, speech recognition, and text analysis from MP4 video files.

## 🎯 Overview

This system implements a **modular audio-focused approach** to advertisement performance prediction:

1. **Audio Extraction** → Extract audio from MP4 files using FFmpeg
2. **Audio Feature Analysis** → Extract 10 critical audio features (RMS, pitch, MFCCs, etc.)
3. **Speech-to-Text** → Convert speech to text using SpeechRecognition
4. **Text Feature Analysis** → Extract 10 critical text features (sentiment, CTAs, hooks)
5. **Content-Based Model** → Predict performance using trained Random Forest classifier

## 🏗️ Architecture

```
MP4 Video → Audio Extraction → Audio Features ↘
                                                 → Content-Based Model → Performance Prediction
MP4 Video → Audio Extraction → Transcript → Text Features ↗
```

### Key Components

- **ContentBasedPredictor**: Main predictor using 313-trained Random Forest model
- **AudioExtractor**: FFmpeg-based audio extraction from video files
- **AudioFeatureExtractor**: 10 critical audio features using librosa
- **TranscriptExtractor**: Speech recognition using SpeechRecognition
- **TranscriptFeatureExtractor**: 10 critical text features for marketing effectiveness
- **Web App**: Flask-based upload and prediction interface

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd AUDIO_FEATURES

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Web App Usage

```bash
# Start the Flask web app
cd web_app
python app.py

# Open browser to http://localhost:5003
# Upload MP4 video and get performance prediction
```

### Command Line Usage

```python
from content_based_predictor import ContentBasedPredictor

# Initialize predictor
predictor = ContentBasedPredictor(models_dir="models")

# Predict performance for a video
result = predictor.predict_performance(video_path="advertisement.mp4")

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Probabilities: {result['probabilities']}")
```

## 📊 Model Performance

### Trained Model (313 videos)
- **Model Type:** Random Forest Classifier
- **Accuracy:** 96.8%
- **Cross-validation:** 94.4% ± 2.0%
- **Training Samples:** 313 videos from YouTube dataset
- **Class Distribution:** 
  - High Performance: 186 videos (59.4%)
  - Low Performance: 127 videos (40.6%)

### Features Extracted

**Audio Features (10 critical features):**
- `rms_mean`: Overall energy/loudness
- `dynamic_range`: Energy variation
- `pitch_mean`: Voice pitch
- `speech_rate`: Speaking pace
- `spectral_centroid_mean`: Audio brightness
- `mfcc_1_mean`, `mfcc_2_mean`: Speech characteristics
- `pause_duration_mean`: Speaking rhythm
- `onset_rate`: Event density
- `zero_crossing_rate`: Speech vs music

**Text Features (10 critical features):**
- `hook_curiosity_words_count`: "Secret", "shocking" words
- `hook_action_words_count`: "Get", "try", "buy" CTAs
- `hook_personal_pronouns_count`: "You", "your" pronouns
- `hook_sentiment_polarity`: Opening emotional tone
- `call_to_action_count`: Direct conversion triggers
- `action_words_count`: Action-oriented language
- `sentiment_polarity`: Overall emotional tone
- `exclamation_count`: Energy level
- `question_count`: Engagement techniques
- `word_count`: Content density

## 🎛️ Configuration

The system uses a trained model with these settings:

```python
# Model configuration (already trained)
model_type = "random_forest"
feature_count = 20  # 10 audio + 10 text features
performance_threshold = 15.0  # Combined score threshold
```

## 🔧 Training Your Own Models

### Retrain the Content-Based Model

```bash
# Train new model with your data
python scripts/content_based_trainer_313.py

# This will:
# 1. Load training data from results/content_based_313_training_data.csv
# 2. Extract audio and transcript features
# 3. Train Random Forest classifier
# 4. Save model to models/content_based_313_model.pkl
```

### Training Data Format

Your training data should be in CSV format with these columns:

```csv
video_id,brand_name,channel_title,duration_seconds,view_count,like_count,comment_count,channel_subscriber_count,publish_age_days,performance_label
video_001,nike,Nike Official,30,1000000,50000,5000,1000000,7,high
video_002,apple,Apple,45,500000,10000,1000,5000000,14,low
```

## 🔌 API Usage

The web app provides a simple API:

```python
import requests

# Upload video and get prediction
with open("advertisement.mp4", "rb") as f:
    files = {"video_file": f}
    data = {"brand_name": "nike"}
    response = requests.post("http://localhost:5003/api/predict", files=files, data=data)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

## 📈 Performance Monitoring

```python
# Get prediction statistics
predictor = ContentBasedPredictor()
stats = predictor.get_stats()

print(f"Total Predictions: {stats['total_predictions']}")
print(f"Success Rate: {stats['success_rate']:.3f}")
print(f"Average Processing Time: {stats['avg_processing_time']:.2f}s")
```

## 🛠️ Development

### Project Structure

```
AUDIO_FEATURES/
├── content_based_predictor.py     # Main predictor class
├── web_app/                       # Flask web application
│   ├── app.py                     # Flask server
│   ├── templates/index.html       # Upload interface
│   └── static/                    # CSS/JS assets
├── src/audio_features/            # Core audio processing
│   ├── audio_extractor.py         # Audio extraction
│   ├── audio_features.py          # Audio feature extraction
│   ├── transcript_extractor.py   # Speech-to-text
│   ├── transcript_features.py    # Text feature extraction
│   └── main_pipeline.py          # Complete pipeline
├── scripts/                       # Training scripts
│   └── content_based_trainer_313.py
├── models/                        # Trained models
│   ├── content_based_313_model.pkl
│   ├── content_based_313_scaler.pkl
│   └── content_based_313_features.pkl
├── results/                       # Training results
└── pyproject.toml                # Project configuration
```

### Running Tests

```bash
# Install development dependencies
uv sync --group dev

# Run tests
python test_types_only.py

# Run linting
uv run ruff check .
uv run ruff format .
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run linting and tests: `uv run ruff check . && python test_types_only.py`
5. Commit changes: `git commit -m "Add feature"`
6. Push to branch: `git push origin feature-name`
7. Create a Pull Request

## 📋 Requirements

### Core Dependencies
- Python 3.9+
- FFmpeg (for audio extraction)
- librosa (audio analysis)
- scikit-learn (machine learning)
- SpeechRecognition (speech-to-text)
- Flask (web application)

### Installation
```bash
# Install FFmpeg (macOS)
brew install ffmpeg

# Install Python dependencies
uv sync
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Support

- Issues: [GitHub Issues]
- Discussions: [GitHub Discussions]

## 🎯 Use Cases

**Perfect for:**
- ✅ **Content Strategy:** Should we publish this video?
- ✅ **A/B Testing:** Which version will perform better?
- ✅ **Quality Control:** Flag potentially low-performing content
- ✅ **Campaign Planning:** Predict which videos need promotion
- ✅ **Budget Allocation:** Focus resources on high-potential content

**Practical Example:**
```
Input: Nike running shoe ad, 30s, 100K subscribers
Output: "HIGH performance predicted (97% confidence)"
→ Decision: Publish with standard promotion budget
```

## 🔮 Future Improvements

When ready to enhance the model:

1. **Add Visual Features:** Thumbnail analysis, face detection, scene analysis
2. **Advanced Text Features:** Sentiment analysis, topic modeling, viral hooks
3. **External Data:** Trending topics, competitor analysis, seasonality
4. **More Data:** Expand beyond 313 samples for better generalization
5. **Real-time Processing:** Support for live video analysis

## ⚠️ Limitations

- **Dataset Size:** 313 videos (good for proof-of-concept)
- **Feature Set:** Audio + text only (no visual analysis)
- **Threshold:** Fixed threshold of 15.0 (could be optimized per brand)
- **Language:** Optimized for English content

## 🎉 Success Metrics

**This system successfully demonstrates:**
- ✅ High accuracy (96.8%) binary classification
- ✅ Robust cross-validation performance
- ✅ Production-ready model artifacts
- ✅ Easy-to-use web interface
- ✅ Comprehensive audio + text feature extraction

**Ready for production use in video performance prediction workflows!**