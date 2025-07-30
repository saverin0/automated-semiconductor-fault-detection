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
import hashlib

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
        self.expected_hashes = {}  # Store expected hashes for model files
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
                # WARNING: Deserializing models with joblib.load can be unsafe. Only load trusted files.
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
        if self.logger:
            self.logger.info(f"🔍 BACKEND: Received DataFrame shape: {df.shape}")
            self.logger.info(f"🔍 BACKEND: Received columns (first 5): {list(df.columns[:5])}")
            self.logger.info(f"🔍 BACKEND: Received columns (last 5): {list(df.columns[-5:])}")
        
        # Always treat the first column as wafer ID
        wafer_ids = df.iloc[:, 0].copy()
        df_features = df.iloc[:, 1:].copy()
        
        if self.logger:
            self.logger.info(f"🔍 BACKEND: After splitting - wafer_ids shape: {wafer_ids.shape}")
            self.logger.info(f"🔍 BACKEND: After splitting - features shape: {df_features.shape}")
        
        # STRICT VALIDATION: Must have exactly 590 feature columns (591 total - 1 wafer column)
        EXPECTED_FEATURE_COLUMNS = 590
        if df_features.shape[1] != EXPECTED_FEATURE_COLUMNS:
            error_msg = f"❌ BACKEND REJECTION: Expected {EXPECTED_FEATURE_COLUMNS} feature columns but got {df_features.shape[1]}. Input had {df.shape[1]} total columns. Cannot proceed with prediction on incomplete data."
            if self.logger:
                self.logger.error(error_msg)
                self.logger.error(f"🔍 BACKEND: Original DataFrame shape: {df.shape}")
                self.logger.error(f"🔍 BACKEND: Features after wafer removal: {df_features.shape}")
            raise ValueError(error_msg)
        
        # Standardize column names: lowercase and replace dashes with underscores
        original_feature_columns = df_features.columns.tolist()
        df_features.columns = [col.lower().replace('-', '_') for col in df_features.columns]
        
        if self.logger:
            self.logger.info(f"🔍 BACKEND: Column standardization - before: {len(original_feature_columns)}, after: {len(df_features.columns)}")
        
        # Get expected feature columns from the model
        model_features = []
        for model in self.models.values():
            if hasattr(model, 'feature_names_in_'):
                model_features = list(model.feature_names_in_)
                # Standardize model feature names - CRITICAL PART
                model_features = [col.lower().replace('-', '_') for col in model_features]
                break
    
        if self.logger:
            self.logger.info(f"✅ BACKEND: Feature validation passed: {len(df_features.columns)} input features match expected {len(model_features)} model features")
        
        # STRICT MAPPING: Use exact positional mapping, no padding with NaN
        if len(df_features.columns) != len(model_features):
            error_msg = f"❌ BACKEND ALIGNMENT ERROR: Feature count mismatch. Input: {len(df_features.columns)}, Model expects: {len(model_features)}"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create DataFrame with the EXACT columns expected by the model
        aligned_df = pd.DataFrame(index=df.index, columns=model_features)
    
        # Fill with values from the input DataFrame by position (1:1 mapping)
        for i, col in enumerate(model_features):
            aligned_df[col] = df_features.iloc[:, i].values
    
        if self.logger:
            self.logger.info(f"✅ BACKEND: Feature alignment successful: {aligned_df.shape}")
            self.logger.info(f"🔍 BACKEND: Aligned DataFrame columns: {len(aligned_df.columns)}")
    
        return aligned_df, wafer_ids

    def handle_problematic_columns(self, X):
        """Attempt to fix problematic columns before imputation."""
        if self.logger:
            self.logger.info("🔧 ATTEMPTING AUTOMATIC DATA REPAIR...")
        
        fixed_issues = []
        
        # 1. Fix zero-variance columns by adding small random noise
        zero_var_cols = []
        for col in X.columns:
            if X[col].var() == 0:
                original_val = X[col].iloc[0] if not X[col].isnull().all() else 0
                # Add tiny random noise (0.1% of value or 0.001 if value is 0)
                noise_scale = max(abs(original_val) * 0.001, 0.001)
                X[col] = original_val + np.random.normal(0, noise_scale, len(X))
                zero_var_cols.append(col)
        
        if zero_var_cols:
            fixed_issues.append(f"Added variance to {len(zero_var_cols)} zero-variance columns")
            if self.logger:
                self.logger.info(f"🔧 Fixed {len(zero_var_cols)} zero-variance columns by adding small noise")
        
        # 2. Fix all-NaN columns by filling with column median from available data
        # If no data available, use 0 as fallback
        all_nan_cols = X.columns[X.isnull().all()].tolist()
        for col in all_nan_cols:
            # Try to use overall dataset median, fallback to 0
            X[col] = 0  # Simple fallback - could be enhanced with training data statistics
        
        if all_nan_cols:
            fixed_issues.append(f"Filled {len(all_nan_cols)} all-NaN columns with fallback values")
            if self.logger:
                self.logger.info(f"🔧 Fixed {len(all_nan_cols)} all-NaN columns with fallback values")
        
        # 3. Handle extremely sparse columns (>95% missing) by forward/backward fill + median
        sparse_cols = []
        for col in X.columns:
            missing_pct = (X[col].isnull().sum() / len(X)) * 100
            if missing_pct > 95:
                # Forward fill, then backward fill, then median fill
                X[col] = X[col].fillna(method='ffill').fillna(method='bfill').fillna(X[col].median()).fillna(0)
                sparse_cols.append(col)
        
        if sparse_cols:
            fixed_issues.append(f"Repaired {len(sparse_cols)} extremely sparse columns")
            if self.logger:
                self.logger.info(f"🔧 Repaired {len(sparse_cols)} extremely sparse columns")
        
        if fixed_issues:
            if self.logger:
                self.logger.info(f"✅ DATA REPAIR COMPLETED: {'; '.join(fixed_issues)}")
        
        return X

    def impute_missing_values(self, X):
        """Impute missing values in prediction data."""
        if self.logger:
            null_counts = X.isnull().sum()
            total_nulls = null_counts.sum()
            self.logger.info(f"🔍 IMPUTATION: Input shape: {X.shape}")
            self.logger.info(f"🔍 IMPUTATION: Total missing values before imputation: {total_nulls}")
            self.logger.info(f"🔍 IMPUTATION: Input columns count: {len(X.columns)}")

        try:
            # Comprehensive data quality analysis before imputation
            original_columns = X.columns.tolist()
            issues = []
            
            # 1. Check for columns with zero variance (constant values)
            zero_var_cols = []
            for col in X.columns:
                if X[col].var() == 0:
                    unique_vals = X[col].dropna().unique()
                    zero_var_cols.append(f"{col} (constant value: {unique_vals[0] if len(unique_vals) > 0 else 'N/A'})")
            
            if zero_var_cols:
                issues.append(f"Zero-variance columns ({len(zero_var_cols)}): {zero_var_cols[:3]}{'...' if len(zero_var_cols) > 3 else ''}")
                if self.logger:
                    self.logger.warning(f"🚨 Found {len(zero_var_cols)} zero-variance columns (constant values)")
            
            # 2. Check for columns that are all NaN
            all_nan_cols = X.columns[X.isnull().all()].tolist()
            if all_nan_cols:
                issues.append(f"All-NaN columns ({len(all_nan_cols)}): {all_nan_cols[:3]}{'...' if len(all_nan_cols) > 3 else ''}")
                if self.logger:
                    self.logger.warning(f"🚨 Found {len(all_nan_cols)} all-NaN columns")
            
            # 3. Check for columns with very high missing percentage (>95%)
            high_missing_cols = []
            for col in X.columns:
                missing_pct = (X[col].isnull().sum() / len(X)) * 100
                if missing_pct > 95:
                    high_missing_cols.append(f"{col} ({missing_pct:.1f}% missing)")
            
            if high_missing_cols:
                issues.append(f"Extremely sparse columns (>95% missing) ({len(high_missing_cols)}): {high_missing_cols[:3]}{'...' if len(high_missing_cols) > 3 else ''}")
                if self.logger:
                    self.logger.warning(f"🚨 Found {len(high_missing_cols)} extremely sparse columns")
            
            # 4. Check for columns with infinite values
            inf_cols = []
            for col in X.columns:
                if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    if np.isinf(X[col]).any():
                        inf_count = np.isinf(X[col]).sum()
                        inf_cols.append(f"{col} ({inf_count} infinite values)")
            
            if inf_cols:
                issues.append(f"Columns with infinite values ({len(inf_cols)}): {inf_cols[:3]}{'...' if len(inf_cols) > 3 else ''}")
                if self.logger:
                    self.logger.warning(f"🚨 Found {len(inf_cols)} columns with infinite values")
            
            # 5. Check for non-numeric columns
            non_numeric_cols = []
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype == 'string':
                    non_numeric_cols.append(f"{col} (dtype: {X[col].dtype})")
            
            if non_numeric_cols:
                issues.append(f"Non-numeric columns ({len(non_numeric_cols)}): {non_numeric_cols[:3]}{'...' if len(non_numeric_cols) > 3 else ''}")
                if self.logger:
                    self.logger.warning(f"🚨 Found {len(non_numeric_cols)} non-numeric columns")
            
            # ATTEMPT AUTOMATIC REPAIR IF ISSUES DETECTED
            if issues:
                if self.logger:
                    self.logger.warning("🔧 Data quality issues detected - attempting automatic repair...")
                X = self.handle_problematic_columns(X)
                
            # Apply KNNImputer
            imputer = KNNImputer()
            data_imputed = imputer.fit_transform(X)
            
            if self.logger:
                self.logger.info(f"🔍 IMPUTATION: KNNImputer output shape: {data_imputed.shape}")
                
            # CRITICAL FIX: If KNNImputer still dropped columns after repair, provide detailed explanation
            if data_imputed.shape[1] != len(original_columns):
                dropped_count = len(original_columns) - data_imputed.shape[1]
                
                # Create detailed error message
                error_details = [
                    f"❌ CRITICAL IMPUTATION ERROR: KNNImputer dropped {dropped_count} columns even after automatic repair!",
                    f"📊 Input columns: {len(original_columns)} → Output columns: {data_imputed.shape[1]}",
                    "",
                    "🔍 ORIGINAL ISSUES DETECTED:",
                ]
                
                if issues:
                    for i, issue in enumerate(issues, 1):
                        error_details.append(f"   {i}. {issue}")
                else:
                    error_details.append("   • No obvious data quality issues detected")
                    error_details.append("   • KNNImputer may have detected subtle numerical issues")
                
                error_details.extend([
                    "",
                    "⚠️  AUTOMATIC REPAIR ATTEMPTED BUT FAILED",
                    "   • Tried to fix zero-variance columns with noise injection",
                    "   • Tried to fill all-NaN columns with fallback values",
                    "   • Tried to repair sparse columns with forward/backward fill",
                    "",
                    "💡 THIS FILE HAS SEVERE DATA QUALITY ISSUES:",
                    "   • Too many sensors with identical readings (stuck/faulty)",
                    "   • Too much missing sensor data", 
                    "   • Data corruption or formatting problems",
                    "",
                    "🔧 MANUAL SOLUTION REQUIRED:",
                    "   • Check sensor calibration and functionality",
                    "   • Verify data collection process",
                    "   • Consider re-collecting this wafer data",
                    "   • Ensure all 590 sensors are functioning properly"
                ])
                
                error_msg = "\n".join(error_details)
                
                if self.logger:
                    self.logger.error(error_msg)
                raise ValueError(error_msg)
                
            if self.logger:
                self.logger.info("✅ IMPUTATION: KNNImputer preserved all columns successfully")

            # Ensure the DataFrame maintains the exact same structure
            X_imputed = pd.DataFrame(data_imputed, columns=original_columns, index=X.index)

            if self.logger:
                self.logger.info(f"✅ IMPUTATION: Final imputed DataFrame shape: {X_imputed.shape}")
                self.logger.info(f"🔍 IMPUTATION: Final columns count: {len(X_imputed.columns)}")
                self.logger.info(f"✅ IMPUTATION: Missing values after imputation: {X_imputed.isnull().sum().sum()}")

            return X_imputed
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ IMPUTATION ERROR: {e}")
                self.logger.error(f"🔍 IMPUTATION: Input shape was: {X.shape}")
                self.logger.error(f"🔍 IMPUTATION: Input columns were: {len(X.columns)}")
            raise

    def assign_clusters(self, X):
        """Assign prediction data to clusters."""
        if self.logger:
            self.logger.info(f"🔍 CLUSTERING: Input shape: {X.shape}")
            self.logger.info(f"🔍 CLUSTERING: Input columns count: {len(X.columns)}")
            
        try:
            # LOAD the clusterer that was saved during training
            clusterer_file = os.getenv('KMEANS_CLUSTERER_FILE', 'kmeans_clusterer.joblib')
            clusterer_path = os.path.join(self.model_dir, clusterer_file)
            if os.path.exists(clusterer_path):
                # WARNING: Deserializing models with joblib.load can be unsafe. Only load trusted files.
                self.clusterer = joblib.load(clusterer_path)
                self.logger.info(f"✅ CLUSTERING: Loaded clusterer from {clusterer_path}")
                clusters = self.clusterer.predict(X)
                if self.logger:
                    self.logger.info(f"✅ CLUSTERING: Cluster assignment successful: {len(clusters)} assignments")
            else:
                # Fallback to creating a new one
                self.logger.warning("⚠️ CLUSTERING: No saved clusterer found! Creating new one (less accurate)")
                num_clusters = len(self.models)
                self.clusterer = KMeans(n_clusters=num_clusters, random_state=42)
                clusters = self.clusterer.fit_predict(X)
                
            return clusters
        except Exception as e:
            self.logger.error(f"❌ CLUSTERING ERROR: {e}")
            self.logger.error(f"🔍 CLUSTERING: Input shape was: {X.shape}")
            # Fallback to assigning all to cluster 0
            self.logger.warning("⚠️ CLUSTERING: Fallback - Assigning all samples to cluster 0")
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