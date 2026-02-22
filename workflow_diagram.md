# Application Workflow Diagram

## Advertisement Prediction Engine - Data Flow for Single Prediction

```mermaid
sequenceDiagram
    participant U as User
    participant W as Web Interface<br/>(Flask App)
    participant F as File System<br/>(Temporary Storage)
    participant FF as FFmpeg<br/>(Audio Extraction)
    participant SR as Google Speech<br/>Recognition API
    participant AF as Audio Feature<br/>Extractor (librosa)
    participant TF as Text Feature<br/>Extractor (TextBlob)
    participant M as Random Forest<br/>Model
    participant S3 as DigitalOcean<br/>Spaces (S3)

    %% Upload and Initial Processing
    U->>W: Upload MP4 Advertisement
    W->>F: Save to Temporary File
    W->>W: Validate File Type<br/>(MP4, AVI, MOV, MKV, WEBM)
    
    %% Audio Extraction
    W->>FF: Extract Audio Track
    FF->>FF: Convert to WAV<br/>(16kHz Mono)
    FF-->>W: Return Audio File Path
    
    %% Parallel Processing
    par Audio Feature Extraction
        W->>AF: Process Audio File
        AF->>AF: Extract 10 Audio Features<br/>(RMS, Pitch, MFCC, etc.)
        AF-->>W: Return Audio Features
    and Speech Recognition
        W->>SR: Convert Speech to Text
        SR->>SR: Optimize Recognition Threshold
        SR-->>W: Return Transcript
        W->>TF: Process Transcript
        TF->>TF: Extract 10 Text Features<br/>(Hooks, CTAs, Sentiment, etc.)
        TF-->>W: Return Text Features
    end
    
    %% Model Processing
    W->>W: Combine Features<br/>(20 Total Features)
    W->>W: Normalize Features<br/>(StandardScaler)
    W->>S3: Load Trained Model<br/>(Random Forest)
    S3-->>W: Return Model & Scaler
    W->>M: Predict Performance
    M->>M: Binary Classification<br/>(High/Low)
    M-->>W: Return Prediction +<br/>Confidence + Feature Analysis
    
    %% Response and Cleanup
    W->>F: Delete Temporary Files
    W-->>U: Display Results<br/>(Prediction + Recommendations)
    
    %% Error Handling
    Note over W: Error Handling:<br/>- Invalid file types<br/>- Speech recognition failures<br/>- Model loading errors<br/>- Feature extraction issues
```

## Workflow Stages

### Stage 1: File Upload & Validation
- User uploads MP4 video file
- System validates file type and size
- File saved to temporary storage

### Stage 2: Audio Extraction
- FFmpeg extracts audio track from video
- Converts to 16kHz mono WAV format
- Optimized for speech recognition

### Stage 3: Parallel Feature Extraction
- **Audio Features**: librosa analyzes 10 audio characteristics
- **Text Features**: Google Speech Recognition + TextBlob analyze 10 text characteristics
- Both processes run simultaneously for efficiency

### Stage 4: Model Processing
- Features combined into 20-feature vector
- StandardScaler normalizes features
- Random Forest model loaded from S3 storage
- Binary classification performed

### Stage 5: Response & Cleanup
- Prediction results formatted for user
- Temporary files cleaned up
- Error handling throughout process

## Performance Characteristics

### Processing Time
- **Audio Extraction**: ~2-5 seconds
- **Feature Extraction**: ~3-8 seconds (parallel)
- **Model Prediction**: ~0.1 seconds
- **Total**: ~5-13 seconds per video

### Error Handling
- Invalid file types rejected
- Speech recognition failures handled gracefully
- Model loading errors with fallback
- Feature extraction issues with default values

### Scalability
- Stateless processing (no session storage)
- Temporary file cleanup prevents storage bloat
- S3 model storage enables horizontal scaling
- Parallel processing optimizes throughput
