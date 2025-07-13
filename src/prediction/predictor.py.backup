import os
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from src.utils.path_utils import validate_env_path, ensure_path_exists

load_dotenv()

class WaferPredictor:
    """Class to handle wafer fault prediction using trained models."""
    
    def __init__(self, models_dir: str, output_dir: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the predictor.
        
        Args:
            models_dir: Directory containing trained model files
            output_dir: Directory to save prediction results
            logger: Logger instance
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # Ensure output directory exists
        ensure_path_exists(self.output_dir)
        
        # Load trained models
        self.models = self._load_models()
        self.logger.info(f"Loaded {len(self.models)} trained models")
        
    def _load_models(self) -> Dict[str, Any]:
        """Load all trained model files from the models directory."""
        models = {}
        
        try:
            model_files = list(self.models_dir.glob("model_cluster_*_*.joblib"))
            
            for model_file in model_files:
                try:
                    model = joblib.load(model_file)
                    model_name = model_file.stem  # filename without extension
                    models[model_name] = model
                    self.logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    self.logger.error(f"Failed to load model {model_file}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error loading models from {self.models_dir}: {e}")
            
        return models
    
    def preprocess_prediction_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess prediction data (similar to training preprocessing).
        
        Args:
            data: Input prediction data
            
        Returns:
            Preprocessed data ready for prediction
        """
        try:
            # Handle wafer ID column - check multiple possible column names
            wafer_id_columns = ['Wafer', 'Unnamed: 0', 'wafer_id', 'id']
            wafer_ids = None
            
            for col in wafer_id_columns:
                if col in data.columns:
                    wafer_ids = data[col].copy()
                    data = data.drop(col, axis=1)
                    self.logger.info(f"Found wafer ID column: {col}")
                    break
            
            if wafer_ids is None:
                wafer_ids = data.index
                self.logger.info("Using row index as wafer IDs")
            
            # Handle missing values
            data = data.replace('?', np.nan)
            
            # Remove columns with all null values
            data = data.dropna(axis=1, how='all')
            
            # Fill remaining missing values with mean (for numeric) or mode (for categorical)
            for column in data.columns:
                if data[column].dtype in ['float64', 'int64']:
                    data[column].fillna(data[column].mean(), inplace=True)
                else:
                    data[column].fillna(data[column].mode()[0] if not data[column].mode().empty else 'Unknown', inplace=True)
            
            # Convert all columns to numeric where possible, but be more careful
            for column in data.columns:
                try:
                    # Only convert if all non-null values can be converted to numeric
                    pd.to_numeric(data[column], errors='raise')
                    data[column] = pd.to_numeric(data[column], errors='coerce')
                except (ValueError, TypeError):
                    # If conversion fails, keep as-is and warn
                    self.logger.warning(f"Column {column} contains non-numeric data, keeping as-is")
            
            # Store wafer IDs for later use
            data['wafer_id'] = wafer_ids
            
            self.logger.info(f"Preprocessed data shape: {data.shape}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {e}")
            raise
    
    def cluster_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply clustering to assign data points to clusters (simplified version).
        In a real scenario, you would use the same clustering model used during training.
        
        Args:
            data: Preprocessed data
            
        Returns:
            Data with cluster assignments
        """
        try:
            # For demonstration, we'll use a simple approach
            # In reality, you should load the same clustering model used during training
            
            # Simple k-means clustering (you should load the actual trained clustering model)
            from sklearn.cluster import KMeans
            
            # Extract features (exclude wafer_id)
            features = data.drop('wafer_id', axis=1) if 'wafer_id' in data.columns else data
            
            # Use 3 clusters (adjust based on your training setup)
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(features)
            
            data['cluster'] = clusters
            
            self.logger.info(f"Data clustered into {len(np.unique(clusters))} clusters")
            return data
            
        except Exception as e:
            self.logger.error(f"Error in data clustering: {e}")
            raise
    
    def predict(self, input_file: str, output_filename: Optional[str] = None) -> str:
        """
        Make predictions on input data.
        
        Args:
            input_file: Path to input CSV file with prediction data
            output_filename: Optional custom output filename
            
        Returns:
            Path to the output file with predictions
        """
        try:
            # Load input data
            data = pd.read_csv(input_file)
            self.logger.info(f"Loaded prediction data: {data.shape}")
            
            # Preprocess data
            processed_data = self.preprocess_prediction_data(data)
            
            # Cluster data
            clustered_data = self.cluster_data(processed_data)
            
            # Make predictions for each cluster
            predictions = []
            wafer_ids = clustered_data['wafer_id'].values
            
            for cluster_id in clustered_data['cluster'].unique():
                cluster_mask = clustered_data['cluster'] == cluster_id
                cluster_data = clustered_data[cluster_mask]
                
                # Find appropriate model for this cluster
                model_name = f"model_cluster_{cluster_id}_RandomForest"
                
                if model_name in self.models:
                    model = self.models[model_name]
                    
                    # Prepare features (exclude wafer_id and cluster columns)
                    features = cluster_data.drop(['wafer_id', 'cluster'], axis=1)
                    
                    # Make predictions
                    cluster_predictions = model.predict(features)
                    cluster_probabilities = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
                    
                    # Store results
                    for i, pred in enumerate(cluster_predictions):
                        result = {
                            'wafer_id': cluster_data.iloc[i]['wafer_id'],
                            'cluster': cluster_id,
                            'prediction': int(pred),
                            'prediction_label': 'Good' if pred == 1 else 'Bad',
                            'model_used': model_name
                        }
                        
                        # Add probability if available
                        if cluster_probabilities is not None:
                            result['confidence'] = float(np.max(cluster_probabilities[i]))
                            result['prob_good'] = float(cluster_probabilities[i][1]) if len(cluster_probabilities[i]) > 1 else float(cluster_probabilities[i][0])
                            result['prob_bad'] = float(cluster_probabilities[i][0]) if len(cluster_probabilities[i]) > 1 else 1 - float(cluster_probabilities[i][0])
                        
                        predictions.append(result)
                        
                    self.logger.info(f"Made {len(cluster_predictions)} predictions for cluster {cluster_id}")
                    
                else:
                    self.logger.warning(f"No model found for cluster {cluster_id}, using default prediction")
                    # Default prediction for clusters without trained models
                    for _, row in cluster_data.iterrows():
                        predictions.append({
                            'wafer_id': row['wafer_id'],
                            'cluster': cluster_id,
                            'prediction': 0,  # Default to 'Bad'
                            'prediction_label': 'Bad',
                            'model_used': 'default',
                            'confidence': 0.5
                        })
            
            # Convert to DataFrame and sort by wafer_id
            results_df = pd.DataFrame(predictions)
            results_df = results_df.sort_values('wafer_id')
            
            # Generate output filename
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                input_basename = Path(input_file).stem
                output_filename = f"predictions_{input_basename}_{timestamp}.csv"
            
            output_path = self.output_dir / output_filename
            
            # Save results
            results_df.to_csv(output_path, index=False)
            
            # Log summary
            total_predictions = len(results_df)
            good_predictions = len(results_df[results_df['prediction'] == 1])
            bad_predictions = total_predictions - good_predictions
            
            self.logger.info(f"Prediction completed:")
            self.logger.info(f"  Total wafers: {total_predictions}")
            self.logger.info(f"  Predicted Good: {good_predictions} ({good_predictions/total_predictions*100:.1f}%)")
            self.logger.info(f"  Predicted Bad: {bad_predictions} ({bad_predictions/total_predictions*100:.1f}%)")
            self.logger.info(f"  Results saved to: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            raise

def predict_from_bigquery_data(
    exported_csv_path: str,
    models_dir: str,
    output_dir: str,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Convenience function to predict from BigQuery exported data.
    
    Args:
        exported_csv_path: Path to CSV file exported from BigQuery
        models_dir: Directory containing trained models
        output_dir: Directory to save predictions
        logger: Logger instance
        
    Returns:
        Path to prediction results file
    """
    predictor = WaferPredictor(models_dir, output_dir, logger)
    return predictor.predict(exported_csv_path)