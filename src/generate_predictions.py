import os
import pandas as pd
import numpy as np
import joblib
import logging
import shutil
import glob
import re
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing.training_data_preprocessing import Preprocessor

def setup_logger():
    logger = logging.getLogger('prediction_generator')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    os.makedirs('logs', exist_ok=True)
    fh = logging.FileHandler('logs/prediction_generation.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def ensure_model_paths(logger, original_model_dir="training_model", common_model_dir="models"):
    """
    Ensures model paths exist and models are available in the expected locations.
    
    Args:
        logger: Logger instance
        original_model_dir: Directory where models are originally saved
        common_model_dir: Directory where models should be accessible for all scripts
        
    Returns:
        Path to the best available model to use
    """
    # Create directories if they don't exist
    os.makedirs(original_model_dir, exist_ok=True)
    os.makedirs(common_model_dir, exist_ok=True)
    
    logger.info(f"Checking for models in {original_model_dir} and {common_model_dir}")
    
    # Path for consolidated model
    consolidated_model_path = os.path.join(common_model_dir, "fault_detection_model.pkl")
    
    # Check if consolidated model already exists
    if os.path.exists(consolidated_model_path):
        logger.info(f"Found consolidated model at {consolidated_model_path}")
        return consolidated_model_path
    
    # Look for models in common directory first
    model_paths = glob.glob(os.path.join(common_model_dir, "*.pkl")) + glob.glob(os.path.join(common_model_dir, "*.joblib"))
    
    if model_paths:
        logger.info(f"Found {len(model_paths)} models in common directory")
        # Use the first model as default and copy it to consolidated path
        shutil.copy2(model_paths[0], consolidated_model_path)
        logger.info(f"Created consolidated model at {consolidated_model_path}")
        return consolidated_model_path
    
    # If no models in common dir, look for cluster models in original location
    cluster_model_pattern = os.path.join(original_model_dir, "model_cluster_*")
    cluster_models = glob.glob(cluster_model_pattern)
    
    if cluster_models:
        logger.info(f"Found {len(cluster_models)} cluster models in {original_model_dir}")
        
        # Use the first cluster model
        model_path = cluster_models[0]
        logger.info(f"Using {os.path.basename(model_path)} as default model")
        
        # Copy to common model directory for other scripts
        shutil.copy2(model_path, consolidated_model_path)
        logger.info(f"Created consolidated model at {consolidated_model_path}")
        return consolidated_model_path
    
    # Look for any models in the original directory
    original_models = glob.glob(os.path.join(original_model_dir, "*.pkl")) + glob.glob(os.path.join(original_model_dir, "*.joblib"))
    
    if original_models:
        logger.info(f"Found {len(original_models)} models in {original_model_dir}")
        # Copy first model to common directory
        shutil.copy2(original_models[0], consolidated_model_path)
        logger.info(f"Created consolidated model at {consolidated_model_path}")
        return consolidated_model_path
    
    # If still no models found, search the entire project for models
    logger.warning("No models found in standard locations. Searching entire project...")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    for root, _, files in os.walk(project_root):
        for file in files:
            if file.endswith('.pkl') or file.endswith('.joblib'):
                if 'model' in file.lower():
                    model_path = os.path.join(root, file)
                    logger.info(f"Found model at {model_path}")
                    # Copy to common directory
                    shutil.copy2(model_path, consolidated_model_path)
                    logger.info(f"Created consolidated model at {consolidated_model_path}")
                    return consolidated_model_path
    
    logger.error("No models found anywhere in the project!")
    return None

def load_model(model_path, logger):
    """Load the trained model from disk."""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def generate_predictions(good_dir, output_file, model_path, logger):
    """
    Generate predictions for all files in the good directory
    and save results to output file.
    """
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path, logger)
    if model is None:
        logger.error("Failed to load model. Exiting.")
        return False
    
    logger.info(f"Processing files from {good_dir}")
    dfs = []
    
    # Process each good file
    for filename in os.listdir(good_dir):
        if not filename.endswith('.csv'):
            continue
            
        filepath = os.path.join(good_dir, filename)
        logger.info(f"Processing {filename}")
        
        try:
            # Read data
            df = pd.read_csv(filepath)
            
            # Handle wafer column variants
            wafer_col = None
            for col in ['Wafer', 'wafer', 'Unnamed: 0']:
                if col in df.columns:
                    wafer_col = col
                    break
                    
            if wafer_col is None:
                # Create an index as wafer ID if missing
                df['wafer_id'] = [f"WAFER_{i}" for i in range(len(df))]
                wafer_values = df['wafer_id'].copy()
            else:
                # Save wafer values
                wafer_values = df[wafer_col].copy()
                # Drop wafer column for prediction
                df = df.drop(wafer_col, axis=1)
            
            # Preprocess data for prediction
            preprocessor = Preprocessor()
            
            # Fill missing values
            df = df.fillna(0)  # Simple imputation for prediction
            
            # Make sure the DataFrame has exactly the features the model expects
            if hasattr(model, 'feature_names_in_'):
                # If model stores feature names (sklearn 1.0+)
                expected_features = model.feature_names_in_
                logger.info(f"Model expects {len(expected_features)} features")
                
                # Keep only features the model knows about
                missing_cols = [col for col in expected_features if col not in df.columns]
                extra_cols = [col for col in df.columns if col not in expected_features]
                
                if missing_cols:
                    logger.warning(f"Missing {len(missing_cols)} columns required by model. Adding with zeros.")
                    for col in missing_cols:
                        df[col] = 0
                
                if extra_cols:
                    logger.info(f"Removing {len(extra_cols)} extra columns not used by model")
                
                # Select only the columns the model expects, in the right order
                df = df[expected_features]
            else:
                logger.warning("Model doesn't store feature names. This might cause prediction errors.")
            
            # Make predictions
            y_pred = model.predict(df)
            
            # Create results dataframe
            result_df = pd.DataFrame({
                'Wafer': wafer_values,
                'Prediction': y_pred,
                'Source_File': filename
            })
            
            dfs.append(result_df)
            logger.info(f"Generated {len(y_pred)} predictions for {filename}")
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            continue
    
    if not dfs:
        logger.error("No predictions were generated. Check if files exist in the good directory.")
        return False
        
    # Combine all results
    final_predictions = pd.concat(dfs, ignore_index=True)
    
    # Map numerical predictions to labels
    final_predictions['Prediction_Label'] = final_predictions['Prediction'].map({
        0: 'Good',
        1: 'Faulty'
    })
    
    # Save to CSV
    final_predictions.to_csv(output_file, index=False)
    logger.info(f"Saved {len(final_predictions)} predictions to {output_file}")
    
    # Summary statistics
    fault_count = final_predictions['Prediction'].sum()
    good_count = len(final_predictions) - fault_count
    fault_percent = (fault_count / len(final_predictions)) * 100 if len(final_predictions) > 0 else 0
    
    summary_msg = (
        f"\n{'=' * 60}\n"
        f"Prediction Results Summary:\n"
        f"Total predictions: {len(final_predictions)}\n"
        f"Good wafers: {good_count} ({100 - fault_percent:.1f}%)\n"
        f"Faulty wafers: {fault_count} ({fault_percent:.1f}%)\n"
        f"{'=' * 60}"
    )
    logger.info(summary_msg)
    
    return True

def main():
    logger = setup_logger()
    
    # Paths
    good_dir = "prediction/good"
    output_file = "prediction/final_predictions.csv"
    
    try:
        # Ensure good directory exists
        if not os.path.exists(good_dir):
            logger.error(f"Good directory {good_dir} does not exist. Run validation first.")
            return
            
        # Check for good files
        good_files = [f for f in os.listdir(good_dir) if f.endswith('.csv')]
        if not good_files:
            logger.error(f"No CSV files found in {good_dir}. Run validation first.")
            return
            
        logger.info(f"Found {len(good_files)} files to process")
        
        # Find and ensure model path
        model_path = ensure_model_paths(logger)
        if not model_path:
            logger.error("No model found. Please train a model first.")
            return
        
        # Generate predictions
        success = generate_predictions(good_dir, output_file, model_path, logger)
        
        if success:
            logger.info(f"Prediction process complete. Results saved to {output_file}")
        else:
            logger.error("Prediction process failed.")
            
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()