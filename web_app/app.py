import logging
import os
import sys
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request
from werkzeug.utils import secure_filename

# Add src to path for imports (go up one level to find src directory)
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import from parent directory (now AUDIO_FEATURES)
sys.path.append(str(Path(__file__).parent.parent))
from content_based_predictor import ContentBasedPredictor

app = Flask(__name__)
app.secret_key = "your-secret-key-here"  # Change this in production

# Configure upload settings
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def setup_app_logging():
    """Setup logging for the Flask app."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


logger = setup_app_logging()

# Initialize the predictor
predictor = None


def initialize_predictor():
    """Initialize the advertisement predictor."""
    global predictor
    try:
        # Set up paths relative to parent directory
        parent_dir = Path(__file__).parent.parent
        models_dir = str(parent_dir / "models")

        predictor = ContentBasedPredictor(models_dir=models_dir)
        # For compatibility with existing response handling, mark binary_only
        predictor.binary_only = True
        logger.info("Predictor initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        return False


@app.route("/")
def index():
    """Main page with upload form."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and prediction."""
    if "video" not in request.files:
        flash("No file selected", "error")
        return redirect(request.url)

    file = request.files["video"]
    if file.filename == "":
        flash("No file selected", "error")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Get additional metadata from form
            brand_name = request.form.get("brand_name", "unknown")
            request.form.get("channel_title") or brand_name

            # Create metadata dictionary

            # Make prediction
            if predictor is None:
                if not initialize_predictor():
                    flash("Prediction system not available", "error")
                    return redirect(request.url)

            # Analyze video
            try:
                logger.info(f"Starting analysis for video: {filepath}")
                analysis = predictor.predict_performance(video_path=filepath)
                logger.info(f"Analysis completed successfully: {analysis}")

                # Validate analysis structure
                if not isinstance(analysis, dict):
                    raise ValueError("Analysis result is not a dictionary")

                # In binary-only mode, expect binary_classification instead of predictions
                if getattr(predictor, "binary_only", False):
                    if "binary_classification" not in analysis:
                        raise ValueError("Analysis missing binary_classification")
                else:
                    if (
                        "predictions" not in analysis
                        and "binary_classification" not in analysis
                    ):
                        raise ValueError(
                            "Analysis missing predictions/binary_classification"
                        )

            except Exception as analysis_error:
                logger.error(f"Analysis failed: {analysis_error}")
                logger.error(f"Video path: {filepath}")
                # Clean up uploaded file even if analysis fails
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise analysis_error

            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

            logger.info(f"Returning analysis result: {analysis}")
            logger.info(f"Predictions: {analysis.get('predictions', {})}")
            logger.info(
                f"Performance level: {analysis.get('performance_level', 'Unknown')}"
            )
            return jsonify({"success": True, "analysis": analysis})

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            flash(f"Error processing video: {str(e)}", "error")
            return jsonify({"success": False, "error": str(e)})
    else:
        flash(
            "Invalid file type. Please upload a video file (mp4, avi, mov, mkv, webm)",
            "error",
        )
        return jsonify({"success": False, "error": "Invalid file type"})


@app.route("/api/predict", methods=["POST"])
def predict_api():
    """API endpoint for predictions."""
    try:
        data = request.get_json()

        if not data or "video_data" not in data:
            return jsonify({"error": "No video data provided"}), 400

        video_data = data["video_data"]

        if predictor is None:
            if not initialize_predictor():
                return jsonify({"error": "Prediction system not available"}), 500

        # Extract video path and metadata from video_data
        video_path = video_data.get("video_path", "")
        analysis = predictor.predict_performance(video_path=video_path)

        return jsonify({"success": True, "analysis": analysis})

    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health_check():
    """Health check endpoint."""
    try:
        if predictor is None:
            initialize_predictor()

        return jsonify({"status": "healthy", "predictor_loaded": predictor is not None})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


if __name__ == "__main__":
    # Initialize predictor on startup
    initialize_predictor()

    # Run the app
    app.run(debug=True, host="0.0.0.0", port=5003)
