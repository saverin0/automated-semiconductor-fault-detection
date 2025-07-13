import os
import logging
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import tempfile
from pathlib import Path
from typing import Dict, Any
import traceback

# Import your existing modules
from src.prediction.predictor import WaferPredictor
from src import main as pipeline_main
from src.utils.path_utils import validate_env_path, ensure_path_exists

# Configure Flask app with proper template directory
app = Flask(__name__, 
           template_folder='/app/src/api/templates',
           static_folder='/app/src/api/static')
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize predictor
models_dir = os.getenv("MODEL_SAVE_DIR", "/app/training_model")
output_dir = os.getenv("PREDICTION_OUTPUT_DIR", "/app/prediction/output")

try:
    predictor = WaferPredictor(models_dir, output_dir, logger)
    logger.info("✅ Predictor initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize predictor: {e}")
    predictor = None

@app.route("/")
def root():
    """Root endpoint - API info"""
    return jsonify({
        "service": "Semiconductor Fault Detection API",
        "version": "1.0.0",
        "status": "running",
        "frontend": "/ui",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "train": "/train",
            "status": "/status",
            "frontend": "/ui"
        }
    })

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "predictor": "available" if predictor else "unavailable",
        "timestamp": pd.Timestamp.now().isoformat()
    })

@app.route("/status")
def status():
    """Get system status"""
    try:
        models_loaded = len(predictor.models) if predictor else 0
        
        return jsonify({
            "system": {
                "environment": {
                    "bq_project": os.getenv("BQ_PROJECT", "unknown"),
                    "model_dir": models_dir,
                    "output_dir": output_dir
                }
            },
            "models_loaded": models_loaded,
            "predictor_status": "available" if predictor else "unavailable"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """Predict wafer faults from uploaded CSV file"""
    if not predictor:
        return jsonify({"error": "Predictor not available"}), 503
    
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            # Make prediction
            output_path = predictor.predict(temp_file_path)
            
            # Read results to get summary
            results_df = pd.read_csv(output_path)
            total_wafers = len(results_df)
            
            # Calculate summary statistics
            good_count = len(results_df[results_df['prediction'] == 1])
            bad_count = total_wafers - good_count
            
            # Calculate average confidence if available
            avg_confidence = None
            if 'confidence' in results_df.columns:
                avg_confidence = round(results_df['confidence'].mean() * 100, 1)
            
            return jsonify({
                "status": "success",
                "predictions": output_path,
                "total_wafers": total_wafers,
                "summary": {
                    "good": good_count,
                    "bad": bad_count,
                    "accuracy": avg_confidence
                }
            })
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/download")
def download():
    """Download prediction results file"""
    try:
        file_path = request.args.get('file')
        if not file_path:
            return jsonify({"error": "No file specified"}), 400
        
        # Security check - ensure file is in output directory
        if not file_path.startswith(output_dir):
            return jsonify({"error": "Invalid file path"}), 403
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        return send_file(file_path, as_attachment=True, 
                        download_name="wafer_predictions.csv")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train", methods=["POST"])
def train():
    """Trigger model training (placeholder)"""
    try:
        # This would trigger your training pipeline
        # For now, return a placeholder response
        return jsonify({
            "status": "training started",
            "message": "Model training initiated in background"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ui")
def frontend():
    """Serve the web frontend"""
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)