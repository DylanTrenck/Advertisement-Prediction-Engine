# Model Architecture Diagram

## Advertisement Prediction Engine - Model Architecture

```mermaid
graph TB
    %% Input Layer
    subgraph "Input Processing"
        VIDEO[MP4 Video File<br/>User Upload]
    end

    %% Audio Processing Pipeline
    subgraph "Audio Processing Pipeline"
        FFMPEG[FFmpeg<br/>Audio Extraction]
        AUDIO[WAV Audio<br/>16kHz Mono]
        LIBROSA[librosa<br/>Audio Analysis]
        AUDIO_FEATURES[10 Audio Features<br/>RMS, Pitch, MFCC, etc.]
    end

    %% Text Processing Pipeline
    subgraph "Text Processing Pipeline"
        SPEECH[Google Speech<br/>Recognition API]
        TRANSCRIPT[Text Transcript<br/>Normalized Audio]
        TEXTBLOB[TextBlob + NLTK<br/>Text Analysis]
        TEXT_FEATURES[10 Text Features<br/>Hooks, CTAs, Sentiment, etc.]
    end

    %% Model Processing
    subgraph "Model Processing"
        COMBINE[Feature Combination<br/>20 Total Features<br/>Audio + Text]
        SCALER[StandardScaler<br/>Feature Normalization]
        RF[Random Forest<br/>Binary Classifier<br/>313 Videos Trained]
    end

    %% Output Layer
    subgraph "Output Processing"
        PREDICTION[Binary Prediction<br/>High/Low Performance]
        CONFIDENCE[Confidence Score<br/>Feature Analysis]
        RECOMMENDATIONS[Improvement<br/>Recommendations]
    end

    %% Data Flow Connections
    VIDEO --> FFMPEG
    FFMPEG --> AUDIO
    AUDIO --> LIBROSA
    AUDIO --> SPEECH
    
    LIBROSA --> AUDIO_FEATURES
    SPEECH --> TRANSCRIPT
    TRANSCRIPT --> TEXTBLOB
    TEXTBLOB --> TEXT_FEATURES
    
    AUDIO_FEATURES --> COMBINE
    TEXT_FEATURES --> COMBINE
    
    COMBINE --> SCALER
    SCALER --> RF
    RF --> PREDICTION
    RF --> CONFIDENCE
    RF --> RECOMMENDATIONS

    %% Styling
    classDef input fill:#e1f5fe
    classDef audio fill:#ffebee
    classDef text fill:#e8f5e8
    classDef model fill:#e3f2fd
    classDef output fill:#fff3e0
    
    class VIDEO input
    class FFMPEG,AUDIO,LIBROSA,AUDIO_FEATURES audio
    class SPEECH,TRANSCRIPT,TEXTBLOB,TEXT_FEATURES text
    class COMBINE,SCALER,RF model
    class PREDICTION,CONFIDENCE,RECOMMENDATIONS output
```

## Key Architecture Components

### Audio Processing Pipeline
- **FFmpeg**: Extracts audio from MP4 files
- **librosa**: Analyzes audio characteristics
- **10 Audio Features**: Focus on attention-grabbing qualities

### Text Processing Pipeline  
- **Google Speech Recognition**: Converts audio to text
- **TextBlob + NLTK**: Analyzes text structure and sentiment
- **10 Text Features**: Focus on marketing effectiveness

### Model Architecture
- **Single Random Forest**: Processes 20 combined features
- **StandardScaler**: Normalizes features for consistent processing
- **Binary Classification**: High vs. Low performance prediction

### Performance Metrics
- **Training Data**: 313 videos from diverse brands
- **Accuracy**: 65% cross-validated
- **Validation**: Both within and outside training set
