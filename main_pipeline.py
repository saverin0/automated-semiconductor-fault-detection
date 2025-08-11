#!/usr/bin/env python3
"""
Main Pipeline for Semiconductor Fault Detection
=========================================================

This script orchestrates the complete ML pipeline from training to prediction.

Workflow:
1. Create JSON schema for training data validation
2. Validate training CSV files (regex checks, null columns)
3. Upload good training files to GCP database
4. Export training data from database
5. Preprocess training data and train models
6. Use tuner.py to find best models for each cluster
7. Validate prediction CSV files
8. Upload prediction files to database
9. Export prediction data from database
10. Run predictions using trained models
11. Save prediction results

Usage:
    python main_pipeline.py --mode training
    python main_pipeline.py --mode prediction
    python main_pipeline.py --mode full
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_main_logger():
    """Set up main pipeline logger."""
    logger = logging.getLogger('main_pipeline')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - MAIN - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    logs_dir = os.getenv('LOGS_DIR', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.getenv('MAIN_LOG_FILE', 'main_pipeline.log')
    if os.path.isabs(log_file):
        log_path = log_file
    else:
        log_path = os.path.join(logs_dir, log_file)
    file_handler = logging.FileHandler(log_path)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

class MainPipeline:
    def __init__(self):
        self.logger = setup_main_logger()
        self.training_data_dir = os.getenv('TRAINING_INPUT_DIR', 'data/training')
        self.prediction_data_dir = os.getenv('PREDICTION_INPUT_DIR', 'data/prediction')
        self.export_dir = os.getenv('EXPORTED_DATA_DIR', 'src/exported_data_from_db')
        self.model_dir = os.getenv('MODEL_SAVE_DIR', 'training_model')
        self.results_dir = os.getenv('RESULTS_DIR', 'prediction_results')
        
    def step_1_create_training_schema(self):
        """Step 1: Create JSON schema for training data validation."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: Creating Training JSON Schema")
        self.logger.info("=" * 60)
        
        try:
            from schema.create_training_json_schema import main as create_training_schema
            result = create_training_schema()
            if result == 0:
                self.logger.info("✅ Training schema created successfully")
                return True
            else:
                self.logger.error("❌ Failed to create training schema")
                return False
        except Exception as e:
            self.logger.error(f"❌ Failed to create training schema: {e}")
            return False
    
    def step_2_validate_training_data(self):
        """Step 2: Validate training CSV files."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: Validating Training Data")
        self.logger.info("=" * 60)
        
        try:
            from src.data_validation.training_data_validation import main as validate_training_data
            validate_training_data()
            self.logger.info("✅ Training data validation completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Training data validation error: {e}")
            return False
    
    def step_3_upload_training_to_db(self):
        """Step 3: Upload good training files to GCP database."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: Uploading Training Data to Database")
        self.logger.info("=" * 60)
        
        try:
            from src.data_ingestion.training_good_csv_to_db import upload_training_data
            upload_result = upload_training_data()
            if upload_result:
                self.logger.info("✅ Training data uploaded to database successfully")
                return True
            else:
                self.logger.error("❌ Failed to upload training data to database")
                return False
        except Exception as e:
            self.logger.error(f"❌ Training data upload error: {e}")
            return False
    
    def step_4_export_training_from_db(self):
        """Step 4: Export training data from database."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 4: Exporting Training Data from Database")
        self.logger.info("=" * 60)
        
        try:
            # Check if exported training data exists
            training_export_path = os.path.join(self.export_dir, "training_exported_data.csv")
            if os.path.exists(training_export_path):
                self.logger.info(f"✅ Training data export found: {training_export_path}")
                return True
            else:
                self.logger.error(f"❌ Training data export not found: {training_export_path}")
                self.logger.info("Please ensure training data has been exported from the database")
                return False
        except Exception as e:
            self.logger.error(f"❌ Training data export check error: {e}")
            return False
    
    def step_5_train_models(self):
        """Step 5: Preprocess training data and train models."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 5: Training Models")
        self.logger.info("=" * 60)
        
        try:
            from src.data_preprocessing.training_data_preprocessing import train_models, setup_logger
            training_export_path = os.path.join(self.export_dir, "training_exported_data.csv")
            
            if not os.path.exists(training_export_path):
                self.logger.error(f"❌ Training data file not found: {training_export_path}")
                return False
            
            # Set up training logger
            training_logger = setup_logger()
            
            # Train models with quick mode (balanced performance for main pipeline)
            train_models(training_export_path, training_logger, optimization_mode='quick')
            self.logger.info("✅ Model training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model training error: {e}")
            return False
    
    def step_6_create_prediction_schema(self):
        """Step 6: Create JSON schema for prediction data validation."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 6: Creating Prediction JSON Schema")
        self.logger.info("=" * 60)
        
        try:
            from schema.create_prediction_json_schema import generate_prediction_schema
            output_path = "schema/schema_prediction.json"
            result = generate_prediction_schema(output_path, num_columns=591)
            if result == 0:
                self.logger.info("✅ Prediction schema created successfully")
                return True
            else:
                self.logger.error("❌ Failed to create prediction schema")
                return False
        except Exception as e:
            self.logger.error(f"❌ Failed to create prediction schema: {e}")
            return False
    
    def step_7_validate_prediction_data(self):
        """Step 7: Validate prediction CSV files."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 7: Validating Prediction Data")
        self.logger.info("=" * 60)
        
        try:
            from src.data_validation.prediction_data_validation import main as validate_prediction_data
            validate_prediction_data()
            self.logger.info("✅ Prediction data validation completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Prediction data validation error: {e}")
            return False
    
    def step_8_upload_prediction_to_db(self):
        """Step 8: Upload prediction files to database."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 8: Uploading Prediction Data to Database")
        self.logger.info("=" * 60)
        
        try:
            from src.data_ingestion.prediction_good_csv_to_db import upload_prediction_data
            upload_result = upload_prediction_data()
            if upload_result:
                self.logger.info("✅ Prediction data uploaded to database successfully")
                return True
            else:
                self.logger.error("❌ Failed to upload prediction data to database")
                return False
        except Exception as e:
            self.logger.error(f"❌ Prediction data upload error: {e}")
            return False
    
    def step_9_export_prediction_from_db(self):
        """Step 9: Export prediction data from database."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 9: Exporting Prediction Data from Database")
        self.logger.info("=" * 60)
        
        try:
            # Check if exported prediction data exists
            prediction_export_path = os.path.join(self.export_dir, "prediction_exported_data.csv")
            if os.path.exists(prediction_export_path):
                self.logger.info(f"✅ Prediction data export found: {prediction_export_path}")
                return True
            else:
                self.logger.error(f"❌ Prediction data export not found: {prediction_export_path}")
                self.logger.info("Please ensure prediction data has been exported from the database")
                return False
        except Exception as e:
            self.logger.error(f"❌ Prediction data export check error: {e}")
            return False
    
    def step_10_run_predictions(self):
        """Step 10: Run predictions using trained models."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 10: Running Predictions")
        self.logger.info("=" * 60)
        
        try:
            from src.model_testing.test_predict_data import test_on_prediction_data
            
            prediction_file = os.path.join(self.export_dir, "prediction_exported_data.csv")
            output_file = os.path.join(self.results_dir, "model_predictions.csv")
            
            if not os.path.exists(prediction_file):
                self.logger.error(f"❌ Prediction data file not found: {prediction_file}")
                return False
            
            if not os.path.exists(self.model_dir):
                self.logger.error(f"❌ Model directory not found: {self.model_dir}")
                return False
            
            # Run predictions
            results = test_on_prediction_data(
                prediction_file=prediction_file,
                model_dir=self.model_dir,
                output_file=output_file
            )
            
            self.logger.info("✅ Predictions completed successfully")
            self.logger.info(f"✅ Results saved to: {output_file}")
            
            # Log prediction summary
            good_count = len(results[results['status'] == 'Good'])
            bad_count = len(results[results['status'] == 'Bad'])
            total_count = len(results)
            
            self.logger.info(f"📊 Prediction Summary:")
            self.logger.info(f"   Total wafers: {total_count}")
            self.logger.info(f"   Good wafers: {good_count} ({good_count/total_count*100:.1f}%)")
            self.logger.info(f"   Bad wafers: {bad_count} ({bad_count/total_count*100:.1f}%)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Prediction error: {e}")
            return False
    
    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        self.logger.info("🚀 Starting TRAINING Pipeline")
        start_time = datetime.now()
        
        steps = [
            ("Create Training Schema", self.step_1_create_training_schema),
            ("Validate Training Data", self.step_2_validate_training_data),
            ("Upload Training to DB", self.step_3_upload_training_to_db),
            ("Export Training from DB", self.step_4_export_training_from_db),
            ("Train Models", self.step_5_train_models)
        ]
        
        for step_name, step_func in steps:
            self.logger.info(f"🔄 Executing: {step_name}")
            if not step_func():
                self.logger.error(f"❌ Training pipeline failed at step: {step_name}")
                return False
        
        end_time = datetime.now()
        duration = end_time - start_time
        self.logger.info("🎉 Training pipeline completed successfully!")
        self.logger.info(f"⏱️  Total execution time: {duration}")
        return True
    
    def run_prediction_pipeline(self):
        """Run the complete prediction pipeline."""
        self.logger.info("🚀 Starting PREDICTION Pipeline")
        start_time = datetime.now()
        
        steps = [
            ("Create Prediction Schema", self.step_6_create_prediction_schema),
            ("Validate Prediction Data", self.step_7_validate_prediction_data),
            ("Upload Prediction to DB", self.step_8_upload_prediction_to_db),
            ("Export Prediction from DB", self.step_9_export_prediction_from_db),
            ("Run Predictions", self.step_10_run_predictions)
        ]
        
        for step_name, step_func in steps:
            self.logger.info(f"🔄 Executing: {step_name}")
            if not step_func():
                self.logger.error(f"❌ Prediction pipeline failed at step: {step_name}")
                return False
        
        end_time = datetime.now()
        duration = end_time - start_time
        self.logger.info("🎉 Prediction pipeline completed successfully!")
        self.logger.info(f"⏱️  Total execution time: {duration}")
        return True
    
    def run_full_pipeline(self):
        """Run both training and prediction pipelines."""
        self.logger.info("🚀 Starting FULL Pipeline (Training + Prediction)")
        start_time = datetime.now()
        
        if not self.run_training_pipeline():
            self.logger.error("❌ Full pipeline failed during training phase")
            return False
        
        if not self.run_prediction_pipeline():
            self.logger.error("❌ Full pipeline failed during prediction phase")
            return False
        
        end_time = datetime.now()
        duration = end_time - start_time
        self.logger.info("🎉 Full pipeline completed successfully!")
        self.logger.info(f"⏱️  Total execution time: {duration}")
        return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Automated Semiconductor Fault Detection Pipeline")
    parser.add_argument('--mode', choices=['training', 'prediction', 'full'], 
                       default='full', help='Pipeline mode to run')
    
    args = parser.parse_args()
    
    # Create pipeline instance
    pipeline = MainPipeline()
    
    # Run selected pipeline
    success = False
    if args.mode == 'training':
        success = pipeline.run_training_pipeline()
    elif args.mode == 'prediction':
        success = pipeline.run_prediction_pipeline()
    elif args.mode == 'full':
        success = pipeline.run_full_pipeline()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
