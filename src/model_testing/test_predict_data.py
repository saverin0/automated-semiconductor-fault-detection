import pandas as pd
import numpy as np
import sys
import os
import logging
import joblib
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
import glob
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_utils import setup_logger as create_logger, get_log_file_path, get_file_path, get_dir_path

def setup_logger():
    """Set up a logger with both console and file output."""
    log_file = os.getenv('PREDICTION_TEST_LOG', 'prediction_test.log')
    return create_logger('prediction_test', log_file)

class ModelPredictor:
    def __init__(self, model_dir=None, logger=None):
        self.model_dir = model_dir or os.getenv('MODEL_SAVE_DIR', 'training_model')
        self.logger = logger
        self.models = {}
        self.imputer = None
        self.clusterer = None
        self.load_models()
    
    def load_models(self):
        """Load all trained models from the model directory."""
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        model_files = glob.glob(os.path.join(self.model_dir, f"{os.getenv('MODEL_CLUSTER_PREFIX', 'model_cluster_')}*.joblib"))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {self.model_dir}")
        
        for model_file in model_files:
            # Extract cluster number from filename
            filename = os.path.basename(model_file)
            # Expected format: model_cluster_0_RandomForest.joblib
            parts = filename.split('_')
            if len(parts) >= 3:
                cluster_id = int(parts[2])
                model = joblib.load(model_file)
                self.models[cluster_id] = model
                if self.logger:
                    self.logger.info(f"Loaded model for cluster {cluster_id}: {filename}")
                
                # Feature importance logging
                X_columns = None
                if hasattr(model, 'feature_names_in_'):
                    X_columns = model.feature_names_in_
                elif hasattr(model, 'get_booster'):
                    # For XGBoost models
                    booster = model.get_booster()
                    X_columns = booster.feature_names
                
                if X_columns is not None and hasattr(model, 'feature_importances_'):
                    importance = pd.DataFrame({
                        'feature': X_columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    self.logger.info(f"Top features for cluster {cluster_id}:")
                    self.logger.info(importance.head(10).to_string(index=False))
        
        if self.logger:
            self.logger.info(f"Total models loaded: {len(self.models)}")
            self.logger.info(f"Available clusters: {list(self.models.keys())}")

    def preprocess_prediction_data(self, df):
        """Preprocess prediction data similar to training preprocessing."""
        # Always treat the first column as wafer ID
        wafer_ids = df.iloc[:, 0].copy()
        df_features = df.iloc[:, 1:].copy()
        
        # Standardize column names: lowercase and replace dashes with underscores
        df_features.columns = [col.lower().replace('-', '_') for col in df_features.columns]
        
        # Get expected feature columns from the model
        model_features = []
        for model in self.models.values():
            if hasattr(model, 'feature_names_in_'):
                model_features = list(model.feature_names_in_)
                # Standardize model feature names - CRITICAL PART
                model_features = [col.lower().replace('-', '_') for col in model_features]
                break
    
        if self.logger:
            self.logger.info(f"Input columns: {len(df_features.columns)}, Model features: {len(model_features)}")
        
        # Create empty DataFrame with the EXACT columns expected by the model
        aligned_df = pd.DataFrame(index=df.index, columns=model_features)
    
        # Fill with values from the input DataFrame where possible, by position not name
        for i, col in enumerate(model_features):
            if i < df_features.shape[1]:
                aligned_df[col] = df_features.iloc[:, i].values
            else:
                aligned_df[col] = np.nan
    
        if self.logger:
            self.logger.info(f"Aligned DataFrame shape: {aligned_df.shape}")
            missing_count = aligned_df.isna().sum().sum()
            self.logger.info(f"Missing values to be imputed: {missing_count}")
    
        return aligned_df, wafer_ids

    def impute_missing_values(self, X):
        """Impute missing values in prediction data."""
        if self.logger:
            null_counts = X.isnull().sum()
            total_nulls = null_counts.sum()
            self.logger.info(f"Total missing values before imputation: {total_nulls}")

        if self.imputer is None:
            self.imputer = KNNImputer()
            data_imputed = self.imputer.fit_transform(X)  # <-- use X here
            if self.logger:
                self.logger.info("Created and fitted new KNNImputer for prediction data.")
        else:
            data_imputed = self.imputer.transform(X)  # <-- use X here
            if self.logger:
                self.logger.info("Used pre-trained imputer for prediction data.")

        X = pd.DataFrame(data_imputed, columns=X.columns, index=X.index)

        if self.logger:
            self.logger.info(f"Missing values after imputation: {X.isnull().sum().sum()}")

        return X

    def assign_clusters(self, X):
        """Assign prediction data to clusters."""
        try:
            # LOAD the clusterer that was saved during training
            clusterer_file = os.getenv('KMEANS_CLUSTERER_FILE', 'kmeans_clusterer.joblib')
            clusterer_path = os.path.join(self.model_dir, clusterer_file)
            if os.path.exists(clusterer_path):
                self.clusterer = joblib.load(clusterer_path)
                self.logger.info(f"Loaded clusterer from {clusterer_path}")
                clusters = self.clusterer.predict(X)
            else:
                # Fallback to creating a new one
                self.logger.warning("No saved clusterer found! Creating new one (less accurate)")
                num_clusters = len(self.models)
                self.clusterer = KMeans(n_clusters=num_clusters, random_state=42)
                clusters = self.clusterer.fit_predict(X)
                
            return clusters
        except Exception as e:
            self.logger.error(f"Error in cluster assignment: {e}")
            # Fallback to assigning all to cluster 0
            self.logger.warning("Fallback: Assigning all samples to cluster 0")
            return np.zeros(len(X))

    def predict(self, df):
        """Make predictions on new data."""
        # Add timestamp at the start of prediction
        processing_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        if self.logger:
            self.logger.info("="*50)
            self.logger.info(f"STARTING PREDICTION PROCESS AT {processing_timestamp}")
            self.logger.info("="*50)
        
        # Preprocess data
        X, wafer_ids = self.preprocess_prediction_data(df.copy())
        
        # Impute missing values
        X_imputed = self.impute_missing_values(X)
        
        # Assign to clusters
        clusters = self.assign_clusters(X_imputed)
        
        # Make predictions for each cluster
        predictions = np.zeros(len(X_imputed))
        prediction_probs = np.zeros(len(X_imputed))
        
        for cluster_id, model in self.models.items():
            cluster_mask = clusters == cluster_id
            cluster_count = cluster_mask.sum()
            
            if cluster_count > 0:
                cluster_X = X_imputed[cluster_mask]
                
                # Make predictions
                cluster_predictions = model.predict(cluster_X)
                predictions[cluster_mask] = cluster_predictions
                
                # Get prediction probabilities if available
                if hasattr(model, 'predict_proba'):
                    cluster_probs = model.predict_proba(cluster_X)
                    # Take probability of positive class
                    prediction_probs[cluster_mask] = cluster_probs[:, 1] if cluster_probs.shape[1] > 1 else cluster_probs[:, 0]
                else:
                    prediction_probs[cluster_mask] = cluster_predictions
                
                if self.logger:
                    self.logger.info(f"Cluster {cluster_id}: {cluster_count} samples predicted")
                    self.logger.info(f"Cluster {cluster_id} predictions: {np.unique(cluster_predictions, return_counts=True)}")
        
        # Create results DataFrame - add timestamp
        results = pd.DataFrame({
            'wafer_id': wafer_ids if wafer_ids is not None else range(len(predictions)),
            'cluster': clusters,
            'prediction': predictions.astype(int),
            'prediction_proba': prediction_probs,
            'status': ['Good' if prob < 0.2 else 'Bad' for prob in prediction_probs],
            'processed_at': processing_timestamp  # Add the timestamp to results
        })
        
        if self.logger:
            self.logger.info("="*50)
            self.logger.info("PREDICTION RESULTS SUMMARY")
            self.logger.info("="*50)
            self.logger.info(f"Total samples processed: {len(results)}")
            self.logger.info(f"Prediction distribution: {results['status'].value_counts().to_dict()}")
            self.logger.info(f"Average prediction probability: {results['prediction_proba'].mean():.3f}")
        
        return results

    def get_model_features(self):
        """Return the list of features expected by the models."""
        model_features = []
        for model in self.models.values():
            if hasattr(model, 'feature_names_in_'):
                model_features = list(model.feature_names_in_)
                # Standardize model feature names
                model_features = [col.lower().replace('-', '_') for col in model_features]
                break
    
        if self.logger:
            self.logger.info(f"Model expects {len(model_features)} features")
        
        return model_features

def test_on_prediction_data(prediction_file=None, model_dir=None, output_file=None):
    """Test trained models on prediction data."""
    logger = setup_logger()
    
    # Use environment variables for default paths
    if prediction_file is None:
        exported_dir = os.getenv('EXPORTED_DATA_DIR', 'src/exported_data_from_db')
        prediction_exported_file = os.getenv('PREDICTION_EXPORTED_FILE', 'prediction_exported_data.csv')
        prediction_file = os.path.join(exported_dir, prediction_exported_file)
    
    if model_dir is None:
        model_dir = os.getenv('MODEL_SAVE_DIR', 'training_model')
    
    if output_file is None:
        results_dir = os.getenv('RESULTS_DIR', 'prediction_results')
        model_predictions_file = os.getenv('MODEL_PREDICTIONS_FILE', 'model_predictions.csv')
        output_file = os.path.join(results_dir, model_predictions_file)
    
    try:
        # Load prediction data
        logger.info(f"Loading prediction data from: {prediction_file}")
        df = pd.read_csv(prediction_file)
        logger.info(f"Prediction data loaded. Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Total wafers in uploaded file: {df.shape[0]}")
        
        # Initialize predictor
        predictor = ModelPredictor(model_dir=model_dir, logger=logger)
        
        # Make predictions
        results = predictor.predict(df)
        
        # Save results if output file specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            results.to_csv(output_file, index=False)
            logger.info(f"Results saved to: {output_file}")
        
        # Display sample results
        logger.info("Sample predictions:")
        logger.info(results.head(10).to_string())
        
        return results
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Configuration from environment variables
    exported_dir = os.getenv('EXPORTED_DATA_DIR', 'src/exported_data_from_db')
    prediction_exported_file = os.getenv('PREDICTION_EXPORTED_FILE', 'prediction_exported_data.csv')
    prediction_file = os.path.join(exported_dir, prediction_exported_file)
    
    model_dir = os.getenv('MODEL_SAVE_DIR', 'training_model')
    
    results_dir = os.getenv('RESULTS_DIR', 'prediction_results')
    model_predictions_file = os.getenv('MODEL_PREDICTIONS_FILE', 'model_predictions.csv')
    output_file = os.path.join(results_dir, model_predictions_file)
    
    # Setup logger
    logger = setup_logger()
    
    # Check if files exist
    if not os.path.exists(prediction_file):
        logger.error(f"Prediction file not found: {prediction_file}")
        logger.info("Please run the prediction data pipeline first to generate prediction data.")
        sys.exit(1)
    
    if not os.path.exists(model_dir):
        logger.error(f"Model directory not found: {model_dir}")
        logger.info("Please run the training pipeline first to generate models.")
        sys.exit(1)
    
    # Run predictions
    try:
        results = test_on_prediction_data(
            prediction_file=prediction_file,
            model_dir=model_dir,
            output_file=output_file
        )
        logger.info("Prediction testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction testing failed: {e}")
        sys.exit(1)