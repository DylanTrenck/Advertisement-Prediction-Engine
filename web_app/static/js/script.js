// Global variables
let selectedFile = null;
let isProcessing = false;
let progressIntervalId = null;
let currentProgress = 0;

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const videoFileInput = document.getElementById('videoFile');
const predictBtn = document.getElementById('predictBtn');
const brandNameInput = document.getElementById('brandName');
const channelTitleInput = document.getElementById('channelTitle');

// Section elements
const uploadSection = document.querySelector('.upload-section');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeDragAndDrop();
    initializeFileInput();
    initializeFormValidation();
});

// Initialize drag and drop functionality
function initializeDragAndDrop() {
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => videoFileInput.click());
}

// Initialize file input
function initializeFileInput() {
    videoFileInput.addEventListener('change', handleFileSelect);
}

// Initialize form validation
function initializeFormValidation() {
    const inputs = [videoFileInput, brandNameInput];
    inputs.forEach(input => {
        input.addEventListener('change', validateForm);
        input.addEventListener('input', validateForm);
    });
}

// Handle drag over
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

// Handle drag leave
function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

// Handle drop
function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// Handle file
function handleFile(file) {
    // Validate file type
    const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/webm'];
    if (!allowedTypes.includes(file.type)) {
        showError('Please select a valid video file (MP4, AVI, MOV, MKV, or WebM)');
        return;
    }
    
    // Validate file size (100MB max)
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (file.size > maxSize) {
        showError('File size must be less than 100MB');
        return;
    }
    
    selectedFile = file;
    displayFileInfo(file);
    validateForm();
}

// Display file information
function displayFileInfo(file) {
    const fileSize = formatFileSize(file.size);
    
    // Update upload area
    uploadArea.innerHTML = `
        <div class="file-info">
            <i class="fas fa-video"></i>
            <div>
                <div class="file-name">${file.name}</div>
                <div class="file-size">${fileSize}</div>
            </div>
        </div>
        <button class="upload-btn" onclick="document.getElementById('videoFile').click()">
            <i class="fas fa-upload"></i> Change Video
        </button>
    `;
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Validate form
function validateForm() {
    const isValid = selectedFile && 
                   brandNameInput.value.trim();
    
    predictBtn.disabled = !isValid;
}

// Handle predict button click
predictBtn.addEventListener('click', async function() {
    if (isProcessing) return;
    
    try {
        isProcessing = true;
        showLoading();
        startProgressTicker('Uploading & analyzing...');
        
        // Create form data
        const formData = new FormData();
        formData.append('video', selectedFile);
        formData.append('brand_name', brandNameInput.value.trim());
        // channel_title intentionally omitted; backend will default to brand name

        // Use XMLHttpRequest to reflect real upload progress
        const result = await new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/upload');

            // We rely on a steady progress ticker for smooth UX; no per-byte updates

            xhr.onload = () => {
                try {
                    const json = JSON.parse(xhr.responseText);
                    stopProgressTicker(100, 'Analysis complete!');
                    resolve(json);
                } catch (e) {
                    reject(e);
                }
            };

            xhr.onerror = () => reject(new Error('Network error'));
            xhr.send(formData);
        });
        console.log('Full response:', result);
        
        if (result.success) {
            setTimeout(() => {
                console.log('Analysis result:', result.analysis);
                showResults(result.analysis);
            }, 500);
        } else {
            console.error('Error result:', result);
            showError(result.error || 'An error occurred while processing the video');
        }
        
    } catch (error) {
        console.error('Error:', error);
        showError('An error occurred while processing the video');
    } finally {
        isProcessing = false;
        stopProgressTicker();
    }
});

// Show loading section
function showLoading() {
    hideAllSections();
    loadingSection.style.display = 'block';
    loadingSection.classList.add('fade-in');
    
    // Reset progress bar
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    progressFill.style.width = '0%';
    progressText.textContent = 'Starting analysis...';
}

// Update progress bar with actual progress
function updateProgress(percentage, message) {
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    
    if (progressFill && progressText) {
        // Clamp to 98% to avoid hitting full early; on complete we set 100% explicitly
        const clamped = Math.min(percentage, 98);
        progressFill.style.width = clamped + '%';
        progressText.textContent = message;
    }
}

// Start a steady progress ticker that advances smoothly until stopped
function startProgressTicker(initialMessage = 'Processing...') {
    stopProgressTicker();
    currentProgress = 0;
    updateProgress(currentProgress, initialMessage);
    // Increment ~0.5% every 150ms → ~3.3%/s, caps at 98%
    progressIntervalId = setInterval(() => {
        // Ease as it approaches 98%
        const remaining = 98 - currentProgress;
        const step = Math.max(0.2, Math.min(0.8, remaining * 0.03));
        currentProgress = Math.min(98, currentProgress + step);
        updateProgress(currentProgress, initialMessage);
    }, 150);
}

// Stop the ticker and optionally set final percent/message
function stopProgressTicker(finalPercent, finalMessage) {
    if (progressIntervalId) {
        clearInterval(progressIntervalId);
        progressIntervalId = null;
    }
    if (typeof finalPercent === 'number') {
        updateProgress(finalPercent, finalMessage || 'Complete');
        const progressFill = document.getElementById('progressFill');
        if (progressFill) progressFill.style.width = '100%';
    }
}

// Show results
function showResults(analysis) {
    try {
        hideAllSections();
        resultsSection.style.display = 'block';
        resultsSection.classList.add('fade-in');
        
        // Progress is now handled by setTimeout, no cleanup needed
        
        console.log('Analysis received:', analysis);
        
        // Get binary classification results
        const binaryClass = analysis.binary_classification || {};
        const prediction = binaryClass.prediction || 'LOW';
        const confidence = binaryClass.confidence || 0;
        const probabilities = binaryClass.probabilities || { high: 0, low: 1 };
        
        // Update main score display
        const scoreElement = document.getElementById('engagementScore');
        const badgeElement = document.getElementById('engagementBadge');
        const badgeTextElement = document.getElementById('engagementBadgeText');
        
        // Show confidence percentage
        const confidencePercent = (confidence * 100).toFixed(1);
        scoreElement.textContent = confidencePercent + '%';
        
        // Update badge
        const isHigh = prediction.toLowerCase() === 'high';
        badgeTextElement.textContent = isHigh ? 'HIGH PERFORMANCE' : 'LOW PERFORMANCE';
        badgeElement.className = `performance-badge ${isHigh ? 'excellent' : 'poor'}`;
        
        // Update context
        const contextElement = document.getElementById('engagementContext');
        const highProb = (probabilities.high * 100).toFixed(1);
        const lowProb = (probabilities.low * 100).toFixed(1);
        contextElement.innerHTML = `
            <strong>Prediction:</strong> ${prediction} performance with ${confidencePercent}% confidence<br>
            <strong>Probabilities:</strong> High: ${highProb}% | Low: ${lowProb}%<br>
            <strong>Based on:</strong> Content analysis (Audio + Transcript features)<br>
            <strong>Model:</strong> RandomForest trained on 313 videos (65.1% accuracy)
        `;
        
        // Update top recommendation
        const topRecommendation = document.getElementById('topRecommendation');
        const recommendations = generateRecommendations(isHigh, confidence, analysis);
        if (recommendations.length > 0) {
            topRecommendation.innerHTML = `
                <i class="fas fa-lightbulb"></i>
                <span>${recommendations[0]}</span>
            `;
        }
        
    } catch (error) {
        console.error('Error showing results:', error);
        showError('Error displaying results: ' + error.message);
    }
}

// Show error
function showError(message) {
    hideAllSections();
    errorSection.style.display = 'block';
    errorSection.classList.add('fade-in');
    
    document.getElementById('errorMessage').textContent = message;
}

// Hide all sections
function hideAllSections() {
    uploadSection.style.display = 'block';
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    
    // Remove fade-in class
    [loadingSection, resultsSection, errorSection].forEach(section => {
        section.classList.remove('fade-in');
    });
}

// Get performance level based on actual training data distribution
function getPerformanceLevel(engagementRate) {
    // Based on analysis of 192 samples from training data:
    // 25th percentile: 0.000615 (0.0615%)
    // 50th percentile: 0.005774 (0.5774%)
    // 75th percentile: 0.014427 (1.4427%)
    // 90th percentile: 0.022243 (2.2243%)
    
    if (engagementRate >= 0.022243) return 'Excellent'; // Top 10%
    if (engagementRate >= 0.014427) return 'Good'; // 75th-90th percentile
    if (engagementRate >= 0.005774) return 'Average'; // 50th-75th percentile
    if (engagementRate >= 0.000615) return 'Below Average'; // 25th-50th percentile
    return 'Very Poor'; // Bottom 25%
}

// Update model predictions
function updateModelPredictions(predictions) {
    const modelsGrid = document.getElementById('modelsGrid');
    modelsGrid.innerHTML = '';
    
    // Display all predicted metrics
    const metrics = [
        { key: 'view_count', label: 'Predicted Views', format: 'number' },
        { key: 'like_count', label: 'Predicted Likes', format: 'number' },
        { key: 'comment_count', label: 'Predicted Comments', format: 'number' },
        { key: 'engagement_rate', label: 'Engagement Rate', format: 'percentage' }
    ];
    
    metrics.forEach((metric, index) => {
        const value = predictions[metric.key] || 0;
        let displayValue;
        
        if (metric.format === 'percentage') {
            displayValue = (value * 100).toFixed(4) + '%';
        } else {
            displayValue = Math.round(value).toLocaleString();
        }
        
        const modelCard = document.createElement('div');
        modelCard.className = 'model-card fade-in';
        modelCard.style.animationDelay = (index * 0.1) + 's';
        modelCard.innerHTML = `
            <div class="model-name">${metric.label}</div>
            <div class="model-score">${displayValue}</div>
        `;
        modelsGrid.appendChild(modelCard);
    });
}

// Update recommendations
function updateRecommendations(recommendations) {
    const recommendationsList = document.getElementById('recommendationsList');
    recommendationsList.innerHTML = '';
    
    if (!Array.isArray(recommendations)) {
        console.warn('Recommendations is not an array:', recommendations);
        return;
    }
    
    recommendations.forEach((recommendation, index) => {
        const recommendationItem = document.createElement('div');
        recommendationItem.className = 'recommendation-item fade-in';
        recommendationItem.style.animationDelay = (index * 0.1) + 's';
        recommendationItem.innerHTML = `
            <i class="fas fa-lightbulb"></i>
            ${recommendation}
        `;
        recommendationsList.appendChild(recommendationItem);
    });
}

// Generate recommendations based on prediction
function generateRecommendations(isHigh, confidence, analysis) {
    const recommendations = [];
    
    // Get feature insights
    const audioFeatures = analysis.audio_features || {};
    const transcriptFeatures = analysis.transcript_features || {};
    
    if (isHigh) {
        if (confidence > 0.8) {
            recommendations.push("🎯 Strong content-based prediction! Your audio and transcript features indicate high performance potential.");
            recommendations.push("🚀 Consider increasing promotion budget - content quality is strong.");
            recommendations.push("📈 Use this video's audio/transcript patterns as a template for future content.");
        } else {
            recommendations.push("✅ Moderate confidence in high performance based on content analysis.");
            recommendations.push("🔄 Consider A/B testing different thumbnails or titles to maximize reach.");
            recommendations.push("📊 Monitor early performance metrics closely.");
        }
    } else {
        if (confidence > 0.8) {
            recommendations.push("⚠️ Content analysis suggests low performance potential.");
            recommendations.push("🎬 Consider improving audio quality or speech clarity.");
            recommendations.push("💡 Try different messaging, call-to-action, or emotional tone.");
        } else {
            recommendations.push("🤔 Uncertain prediction - content features are mixed.");
            recommendations.push("📝 Consider manual review of audio quality and transcript clarity.");
            recommendations.push("🔍 Test with a smaller audience first before full launch.");
        }
    }
    
    // Audio-specific recommendations
    if (audioFeatures.speech_rate !== undefined) {
        if (audioFeatures.speech_rate < 0.5) {
            recommendations.push("🗣️ Consider speaking faster - current speech rate may be too slow for engagement.");
        } else if (audioFeatures.speech_rate > 2.0) {
            recommendations.push("⏸️ Consider slowing down speech - current rate may be too fast for comprehension.");
        }
    }
    
    if (audioFeatures.rms_mean !== undefined && audioFeatures.rms_mean < 0.1) {
        recommendations.push("🔊 Audio energy is low - consider increasing volume or adding background music.");
    }
    
    // Transcript-specific recommendations
    if (transcriptFeatures.word_count !== undefined) {
        if (transcriptFeatures.word_count < 50) {
            recommendations.push("📝 Consider adding more content - transcript is quite short.");
        } else if (transcriptFeatures.word_count > 500) {
            recommendations.push("✂️ Consider shortening content - transcript may be too long for optimal engagement.");
        }
    }
    
    if (transcriptFeatures.sentiment_polarity !== undefined) {
        if (transcriptFeatures.sentiment_polarity < -0.2) {
            recommendations.push("😊 Consider more positive messaging - current sentiment is quite negative.");
        } else if (transcriptFeatures.sentiment_polarity > 0.5) {
            recommendations.push("⚖️ Consider balancing positive messaging with practical information.");
        }
    }
    
    // General video recommendations
    const videoMeta = analysis.video_metadata || {};
    if (videoMeta.duration_seconds > 60) {
        recommendations.push("⏱️ Consider shorter duration for better engagement (current: " + Math.round(videoMeta.duration_seconds) + "s).");
    }
    
    return recommendations;
}

// Update detailed analysis
function updateDetailedAnalysis(analysis) {
    // Brand analysis
    const brandAnalysis = document.getElementById('brandAnalysis');
    const videoMeta = analysis.video_metadata || {};
    brandAnalysis.innerHTML = `
        <strong>Video Details:</strong><br>
        Duration: ${Math.round(videoMeta.duration_seconds || 0)}s<br>
        Resolution: ${videoMeta.width || 'N/A'}x${videoMeta.height || 'N/A'}<br>
        File Size: ${(videoMeta.file_size_mb || 0).toFixed(1)}MB
    `;
    
    // Content analysis
    const contentAnalysis = document.getElementById('contentAnalysis');
    const binaryClass = analysis.binary_classification || {};
    const prediction = binaryClass.prediction || 'UNKNOWN';
    const confidence = ((binaryClass.confidence || 0) * 100).toFixed(1);
    
    // Get audio and transcript features if available
    const audioFeatures = analysis.audio_features || {};
    const transcriptFeatures = analysis.transcript_features || {};
    
    let featureDetails = '';
    if (Object.keys(audioFeatures).length > 0 || Object.keys(transcriptFeatures).length > 0) {
        featureDetails = '<br><strong>Key Features Analyzed:</strong><br>';
        
        // Show top audio features
        if (audioFeatures.rms_mean !== undefined) {
            featureDetails += `• Audio Energy: ${audioFeatures.rms_mean.toFixed(3)}<br>`;
        }
        if (audioFeatures.speech_rate !== undefined) {
            featureDetails += `• Speech Rate: ${audioFeatures.speech_rate.toFixed(2)}<br>`;
        }
        
        // Show top transcript features
        if (transcriptFeatures.word_count !== undefined) {
            featureDetails += `• Word Count: ${transcriptFeatures.word_count}<br>`;
        }
        if (transcriptFeatures.sentiment_polarity !== undefined) {
            featureDetails += `• Sentiment: ${transcriptFeatures.sentiment_polarity.toFixed(2)}<br>`;
        }
    }
    
    contentAnalysis.innerHTML = `
        <strong>Content Analysis:</strong><br>
        Prediction: ${prediction} Performance<br>
        Confidence: ${confidence}%<br>
        Model: RandomForest (313 videos)${featureDetails}
    `;
}

// Reset form
function resetForm() {
    selectedFile = null;
    videoFileInput.value = '';
    brandNameInput.value = '';
    channelTitleInput.value = '';
    predictBtn.disabled = true;
    
    // Reset upload area
    uploadArea.innerHTML = `
        <div class="upload-icon">
            <i class="fas fa-cloud-upload-alt"></i>
        </div>
        <h3>Upload Your Advertisement</h3>
        <p>Drag and drop your video file here or click to browse</p>
        <input type="file" id="videoFile" accept="video/*" style="display: none;">
        <button class="upload-btn" onclick="document.getElementById('videoFile').click()">
            <i class="fas fa-upload"></i> Choose Video
        </button>
    `;
    
    // Reinitialize file input
    initializeFileInput();
    
    hideAllSections();
}

// Utility function to show notifications
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 10px;
        color: white;
        font-weight: 500;
        z-index: 1000;
        animation: slideIn 0.3s ease-out;
        max-width: 300px;
    `;
    
    // Set background color based on type
    if (type === 'error') {
        notification.style.background = 'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)';
    } else if (type === 'success') {
        notification.style.background = 'linear-gradient(135deg, #27ae60 0%, #2ecc71 100%)';
    } else {
        notification.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
    }
    
    // Add to page
    document.body.appendChild(notification);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 5000);
}

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style); 