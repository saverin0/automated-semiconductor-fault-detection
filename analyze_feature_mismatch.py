import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("feature_analyzer")

def analyze_model_features(model_path: str) -> List[str]:
    """Analyze what features a model expects"""
    logger.info(f"Analyzing model: {model_path}")
    model = joblib.load(model_path)
    
    if hasattr(model, 'feature_names_in_'):
        features = model.feature_names_in_
        logger.info(f"✓ Model expects {len(features)} features")
        logger.info(f"✓ First 5 expected features: {features[:5]}")
        return features
    else:
        logger.warning("❌ Model doesn't have feature_names_in_ attribute")
        return []

def analyze_csv_features(csv_path: str) -> Dict:
    """Analyze features in a CSV file"""
    logger.info(f"Analyzing CSV file: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    result = {
        "total_columns": len(df.columns),
        "total_rows": len(df),
        "columns": df.columns.tolist(),
        "first_five": df.columns[:5].tolist()
    }
    
    # Check if there's a wafer ID column
    id_columns = ['Wafer', 'Unnamed: 0', 'wafer_id']
    for col in id_columns:
        if col in df.columns:
            result["id_column"] = col
            result["id_examples"] = df[col].head().tolist()
            break
    
    # Check data types
    result["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Count non-numeric columns
    non_numeric = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    result["non_numeric_columns"] = non_numeric
    
    # Analyze missing values
    result["missing_values"] = df.isnull().sum().sum()
    
    logger.info(f"✓ CSV has {result['total_columns']} columns and {result['total_rows']} rows")
    if "id_column" in result:
        logger.info(f"✓ Found ID column: {result['id_column']} with values like {result['id_examples'][:3]}")
    logger.info(f"✓ First 5 columns: {result['first_five']}")
    logger.info(f"✓ Non-numeric columns: {len(non_numeric)}")
    logger.info(f"✓ Total missing values: {result['missing_values']}")
    
    return result

def compare_features(model_features: List[str], csv_features: List[str]) -> Dict:
    """Compare features between model and CSV"""
    logger.info("Comparing model features with CSV features...")
    
    # Convert lists to sets for comparison
    model_set = set(model_features)
    csv_set = set(csv_features)
    
    # Find differences
    missing_in_csv = model_set - csv_set
    extra_in_csv = csv_set - model_set
    common = model_set.intersection(csv_set)
    
    result = {
        "model_features_count": len(model_features),
        "csv_features_count": len(csv_features),
        "common_features_count": len(common),
        "missing_features_count": len(missing_in_csv),
        "extra_features_count": len(extra_in_csv),
        "missing_features": list(missing_in_csv)[:10],  # First 10 missing features
        "extra_features": list(extra_in_csv)[:10]       # First 10 extra features
    }
    
    logger.info(f"✓ Model expects {len(model_features)} features")
    logger.info(f"✓ CSV has {len(csv_features)} features")
    logger.info(f"✓ Common features: {len(common)}")
    logger.info(f"✓ Missing in CSV: {len(missing_in_csv)} features")
    logger.info(f"✓ Extra in CSV: {len(extra_in_csv)} features")
    
    if missing_in_csv:
        logger.warning(f"❌ Examples of missing features: {list(missing_in_csv)[:5]}")
    if extra_in_csv:
        logger.info(f"ℹ️ Examples of extra features: {list(extra_in_csv)[:5]}")
    
    return result

def create_feature_aligned_data(csv_path: str, model_features: List[str], output_path: Optional[str] = None) -> pd.DataFrame:
    """Create a version of the CSV with features aligned to the model"""
    logger.info(f"Creating feature-aligned data from {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    original_shape = df.shape
    
    # Handle wafer ID column
    id_columns = ['Wafer', 'Unnamed: 0', 'wafer_id']
    wafer_ids = None
    
    for col in id_columns:
        if col in df.columns:
            wafer_ids = df[col].copy()
            logger.info(f"Preserving ID column {col}")
            df = df.drop(col, axis=1)
            break
    
    # Check for missing features required by the model
    missing_features = set(model_features) - set(df.columns)
    if missing_features:
        logger.warning(f"Adding {len(missing_features)} missing features with zeros")
        for feature in missing_features:
            df[feature] = 0
    
    # Select only the features the model expects, in the right order
    aligned_df = df[model_features]
    
    # Add back the wafer IDs if present
    if wafer_ids is not None:
        aligned_df = pd.concat([wafer_ids.rename('wafer_id'), aligned_df], axis=1)
    
    logger.info(f"Original shape: {original_shape}")
    logger.info(f"Aligned shape: {aligned_df.shape}")
    
    # Save to file if requested
    if output_path:
        aligned_df.to_csv(output_path, index=False)
        logger.info(f"Saved aligned data to {output_path}")
    
    return aligned_df

def main():
    # Define paths
    model_paths = [
        "training_model/model_cluster_0_RandomForest.joblib",
        "training_model/model_cluster_1_RandomForest.joblib",
        "training_model/model_cluster_2_RandomForest.joblib"
    ]
    
    prediction_file = "prediction/input/wafer_07012020_041011.csv"
    
    # Step 1: Analyze a model
    logger.info("\n==== ANALYZING MODEL FEATURES ====")
    model_features = []
    for model_path in model_paths:
        if os.path.exists(model_path):
            features = analyze_model_features(model_path)
            if features is not None and len(features) > 0:
                model_features = features
                break
    
    if not model_features:
        logger.error("❌ No model features found! Can't continue.")
        return
    
    # Step 2: Analyze CSV file
    logger.info("\n==== ANALYZING CSV FEATURES ====")
    csv_result = analyze_csv_features(prediction_file)
    
    # Step 3: Compare features
    logger.info("\n==== COMPARING FEATURES ====")
    csv_features = [col for col in csv_result["columns"] 
                    if col not in ['Wafer', 'Unnamed: 0', 'wafer_id']]
    comparison = compare_features(model_features, csv_features)
    
    # Step 4: Create aligned data if needed
    if comparison["missing_features_count"] > 0 or comparison["extra_features_count"] > 0:
        logger.info("\n==== CREATING ALIGNED DATA ====")
        aligned_df = create_feature_aligned_data(
            prediction_file, model_features, "aligned_prediction_data.csv")
        
        logger.info("\n✅ Created aligned prediction data - use this for predictions!")
    else:
        logger.info("\n✅ Features match perfectly - no alignment needed!")

if __name__ == "__main__":
    main()
