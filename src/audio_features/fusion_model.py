"""
Fusion model that combines audio and transcript features for final prediction.
"""

import logging
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .types import (
    AudioFeatures,
    ModelPrediction,
    ModelTrainingConfig,
    TrainingDataPoint,
    TranscriptFeatures,
)


class FusionModel:
    """
    Fusion model that combines audio and transcript features for performance prediction.

    Uses ensemble learning to combine predictions from:
    - Audio-only model
    - Transcript-only model
    - Combined feature model

    Examples:
        >>> model = FusionModel()
        >>> prediction = model.predict(audio_features, transcript_features)
        >>> assert prediction["final_prediction"] in ["high", "low"]
    """

    def __init__(self, model_type: str = "xgboost"):
        """
        Initialize the fusion model.

        Args:
            model_type: Type of model to use ("xgboost", "random_forest", "logistic")
        """
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type

        # Individual models
        self.audio_model = None
        self.transcript_model = None
        self.fusion_model = None

        # Preprocessors
        self.audio_scaler = StandardScaler()
        self.transcript_scaler = StandardScaler()
        self.fusion_scaler = StandardScaler()

        # Feature names for consistency
        self.audio_feature_names = []
        self.transcript_feature_names = []
        self.fusion_feature_names = []

        # Model performance tracking
        self.training_history = []

    def _create_model(self) -> any:
        """Create a model instance based on model_type."""
        if self.model_type == "xgboost":
            return XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss",
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
        elif self.model_type == "logistic":
            return LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(
        self, training_data: list[TrainingDataPoint], config: ModelTrainingConfig
    ) -> dict:
        """
        Train the fusion model on training data.

        Args:
            training_data: List of training data points
            config: Training configuration

        Returns:
            Training results and metrics
        """
        self.logger.info(f"Training fusion model with {len(training_data)} samples")

        # Filter out data points without extracted features
        valid_training_data = []
        for data_point in training_data:
            if (
                data_point.get("audio_features")
                and data_point.get("transcript_features")
                and len(data_point["audio_features"]) > 0
                and len(data_point["transcript_features"]) > 0
            ):
                valid_training_data.append(data_point)

        self.logger.info(
            f"Using {len(valid_training_data)} samples with valid features for training"
        )

        if len(valid_training_data) == 0:
            raise ValueError("No valid training data with extracted features found")

        # Extract features and labels
        audio_features_list = []
        transcript_features_list = []
        labels = []

        for data_point in valid_training_data:
            audio_features_list.append(data_point["audio_features"])
            transcript_features_list.append(data_point["transcript_features"])
            labels.append(1 if data_point["performance_label"] == "high" else 0)

        # Convert to DataFrames
        audio_df = pd.DataFrame(audio_features_list)
        transcript_df = pd.DataFrame(transcript_features_list)

        # Store feature names
        self.audio_feature_names = list(audio_df.columns)
        self.transcript_feature_names = list(transcript_df.columns)

        # Handle missing values
        audio_df = audio_df.fillna(0)
        transcript_df = transcript_df.fillna(0)

        # Create combined features
        fusion_df = pd.concat([audio_df, transcript_df], axis=1)
        self.fusion_feature_names = list(fusion_df.columns)

        # Train-validation split
        X_audio = audio_df.values
        X_transcript = transcript_df.values
        X_fusion = fusion_df.values
        y = np.array(labels)

        # Split data
        (
            X_audio_train,
            X_audio_val,
            X_transcript_train,
            X_transcript_val,
            X_fusion_train,
            X_fusion_val,
            y_train,
            y_val,
        ) = train_test_split(
            X_audio,
            X_transcript,
            X_fusion,
            y,
            test_size=config["validation_split"],
            random_state=config["random_state"],
            stratify=y,
        )

        # Scale features
        X_audio_train_scaled = self.audio_scaler.fit_transform(X_audio_train)
        X_audio_val_scaled = self.audio_scaler.transform(X_audio_val)

        X_transcript_train_scaled = self.transcript_scaler.fit_transform(
            X_transcript_train
        )
        X_transcript_val_scaled = self.transcript_scaler.transform(X_transcript_val)

        X_fusion_train_scaled = self.fusion_scaler.fit_transform(X_fusion_train)
        X_fusion_val_scaled = self.fusion_scaler.transform(X_fusion_val)

        # Train individual models
        self.logger.info("Training audio model...")
        self.audio_model = self._create_model()
        self.audio_model.fit(X_audio_train_scaled, y_train)

        self.logger.info("Training transcript model...")
        self.transcript_model = self._create_model()
        self.transcript_model.fit(X_transcript_train_scaled, y_train)

        self.logger.info("Training fusion model...")
        self.fusion_model = self._create_model()
        self.fusion_model.fit(X_fusion_train_scaled, y_train)

        # Evaluate models
        results = self._evaluate_models(
            X_audio_val_scaled, X_transcript_val_scaled, X_fusion_val_scaled, y_val
        )

        # Cross-validation
        if config["cross_validation_folds"] > 1:
            cv_results = self._cross_validate(X_fusion_train_scaled, y_train, config)
            results.update(cv_results)

        # Store training history
        self.training_history.append(results)

        return results

    def _evaluate_models(self, X_audio, X_transcript, X_fusion, y_true) -> dict:
        """Evaluate all three models on validation data."""
        results = {}

        # Audio model
        y_pred_audio = self.audio_model.predict(X_audio)
        results["audio_accuracy"] = accuracy_score(y_true, y_pred_audio)

        # Transcript model
        y_pred_transcript = self.transcript_model.predict(X_transcript)
        results["transcript_accuracy"] = accuracy_score(y_true, y_pred_transcript)

        # Fusion model
        y_pred_fusion = self.fusion_model.predict(X_fusion)
        results["fusion_accuracy"] = accuracy_score(y_true, y_pred_fusion)

        # Feature importance for fusion model
        if hasattr(self.fusion_model, "feature_importances_"):
            importance_dict = dict(
                zip(
                    self.fusion_feature_names,
                    self.fusion_model.feature_importances_,
                    strict=False,
                )
            )
            # Get top 10 features
            top_features = sorted(
                importance_dict.items(), key=lambda x: x[1], reverse=True
            )[:10]
            results["top_features"] = top_features

        return results

    def _cross_validate(self, X, y, config: ModelTrainingConfig) -> dict:
        """Perform cross-validation."""
        cv_scores = cross_val_score(
            self.fusion_model,
            X,
            y,
            cv=config["cross_validation_folds"],
            scoring="accuracy",
        )

        return {
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores),
            "cv_scores": cv_scores.tolist(),
        }

    def predict(
        self, audio_features: AudioFeatures, transcript_features: TranscriptFeatures
    ) -> ModelPrediction:
        """
        Make prediction using all three models.

        Args:
            audio_features: Audio feature dictionary
            transcript_features: Transcript feature dictionary

        Returns:
            Combined prediction with confidence scores
        """
        import time

        start_time = time.time()

        try:
            # Convert features to arrays
            audio_array = self._features_to_array(
                audio_features, self.audio_feature_names
            )
            transcript_array = self._features_to_array(
                transcript_features, self.transcript_feature_names
            )

            # Combine features
            fusion_array = np.concatenate([audio_array, transcript_array])

            # Scale features
            audio_scaled = self.audio_scaler.transform(audio_array.reshape(1, -1))
            transcript_scaled = self.transcript_scaler.transform(
                transcript_array.reshape(1, -1)
            )
            fusion_scaled = self.fusion_scaler.transform(fusion_array.reshape(1, -1))

            # Get predictions and probabilities
            self.audio_model.predict(audio_scaled)[0]
            audio_proba = self.audio_model.predict_proba(audio_scaled)[0]

            self.transcript_model.predict(transcript_scaled)[0]
            transcript_proba = self.transcript_model.predict_proba(transcript_scaled)[0]

            fusion_pred = self.fusion_model.predict(fusion_scaled)[0]
            fusion_proba = self.fusion_model.predict_proba(fusion_scaled)[0]

            # Extract feature importance for explanation
            top_audio_features = self._get_top_features(audio_features, "audio")
            top_transcript_features = self._get_top_features(
                transcript_features, "transcript"
            )

            return ModelPrediction(
                audio_model_score=float(audio_proba[1]),  # Probability of "high" class
                audio_model_confidence=float(max(audio_proba)),
                transcript_model_score=float(transcript_proba[1]),
                transcript_model_confidence=float(max(transcript_proba)),
                fusion_model_score=float(fusion_proba[1]),
                fusion_model_confidence=float(max(fusion_proba)),
                final_prediction="high" if fusion_pred == 1 else "low",
                top_audio_features=top_audio_features,
                top_transcript_features=top_transcript_features,
                processing_time_seconds=time.time() - start_time,
                model_version=f"{self.model_type}_v1.0",
                prediction_date=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return self._get_empty_prediction()

    def _features_to_array(
        self, features: dict, feature_names: list[str]
    ) -> np.ndarray:
        """Convert feature dictionary to numpy array in correct order."""
        array = np.zeros(len(feature_names))
        for i, name in enumerate(feature_names):
            array[i] = features.get(name, 0.0)
        return array

    def _get_top_features(
        self, features: dict, model_type: str
    ) -> list[tuple[str, float]]:
        """Get top contributing features for explanation."""
        if model_type == "audio" and hasattr(self.audio_model, "feature_importances_"):
            importance = self.audio_model.feature_importances_
            feature_names = self.audio_feature_names
        elif model_type == "transcript" and hasattr(
            self.transcript_model, "feature_importances_"
        ):
            importance = self.transcript_model.feature_importances_
            feature_names = self.transcript_feature_names
        else:
            return []

        # Get feature values and importance
        feature_contributions = []
        for i, name in enumerate(feature_names):
            value = features.get(name, 0.0)
            contrib = abs(value * importance[i])
            feature_contributions.append((name, contrib))

        # Sort by contribution and return top 5
        feature_contributions.sort(key=lambda x: x[1], reverse=True)
        return feature_contributions[:5]

    def _get_empty_prediction(self) -> ModelPrediction:
        """Return empty prediction for error cases."""
        return ModelPrediction(
            audio_model_score=0.5,
            audio_model_confidence=0.0,
            transcript_model_score=0.5,
            transcript_model_confidence=0.0,
            fusion_model_score=0.5,
            fusion_model_confidence=0.0,
            final_prediction="low",
            top_audio_features=[],
            top_transcript_features=[],
            processing_time_seconds=0.0,
            model_version="error",
            prediction_date="",
        )

    def save_models(self, save_dir: str) -> None:
        """Save all trained models and scalers."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save models
        joblib.dump(self.audio_model, save_path / "audio_model.pkl")
        joblib.dump(self.transcript_model, save_path / "transcript_model.pkl")
        joblib.dump(self.fusion_model, save_path / "fusion_model.pkl")

        # Save scalers
        joblib.dump(self.audio_scaler, save_path / "audio_scaler.pkl")
        joblib.dump(self.transcript_scaler, save_path / "transcript_scaler.pkl")
        joblib.dump(self.fusion_scaler, save_path / "fusion_scaler.pkl")

        # Save feature names
        metadata = {
            "audio_feature_names": self.audio_feature_names,
            "transcript_feature_names": self.transcript_feature_names,
            "fusion_feature_names": self.fusion_feature_names,
            "model_type": self.model_type,
            "training_history": self.training_history,
        }

        with open(save_path / "model_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        self.logger.info(f"Models saved to {save_dir}")

    def load_models(self, load_dir: str) -> None:
        """Load trained models and scalers."""
        load_path = Path(load_dir)

        # Load models
        self.audio_model = joblib.load(load_path / "audio_model.pkl")
        self.transcript_model = joblib.load(load_path / "transcript_model.pkl")
        self.fusion_model = joblib.load(load_path / "fusion_model.pkl")

        # Load scalers
        self.audio_scaler = joblib.load(load_path / "audio_scaler.pkl")
        self.transcript_scaler = joblib.load(load_path / "transcript_scaler.pkl")
        self.fusion_scaler = joblib.load(load_path / "fusion_scaler.pkl")

        # Load metadata
        with open(load_path / "model_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        self.audio_feature_names = metadata["audio_feature_names"]
        self.transcript_feature_names = metadata["transcript_feature_names"]
        self.fusion_feature_names = metadata["fusion_feature_names"]
        self.model_type = metadata["model_type"]
        self.training_history = metadata.get("training_history", [])

        self.logger.info(f"Models loaded from {load_dir}")
