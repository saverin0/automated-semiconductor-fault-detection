from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import pandas as pd
import numpy as np
import os
import sys
import joblib
from datetime import datetime
import plotly
import json
import plotly.express as px
from pathlib import Path
import re
from dotenv import load_dotenv
import warnings

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import prediction functionality
from src.model_testing.test_predict_data import ModelPredictor, setup_logger
from src.utils.performance_monitor import monitor_performance

# Define the cleanup function here (before it's used)
def cleanup_old_files(folder_path, max_files=20):
    """Clean up old files, keeping only the most recent max_files"""
    try:
        files = sorted(Path(folder_path).glob("*.*"), key=os.path.getmtime, reverse=True)
        if len(files) > max_files:
            for file_to_delete in files[max_files:]:
                os.remove(file_to_delete)
                logger.info(f"Deleted old file: {file_to_delete}")
        return len(files) - max_files if len(files) > max_files else 0
    except Exception as e:
        logger.error(f"Error cleaning up old files: {e}")
        return 0

app = Flask(__name__)
# Ensure FLASK_SECRET_KEY is set for security
secret_key = os.getenv('FLASK_SECRET_KEY')
if not secret_key:
    raise RuntimeError("FLASK_SECRET_KEY environment variable must be set for security")
app.secret_key = secret_key

# Initialize model predictor
MODEL_DIR = os.getenv('MODEL_SAVE_DIR', 'training_model')
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
RESULTS_FOLDER = os.getenv('RESULTS_DIR', 'prediction_results')

# Setup logger before using it
logger = setup_logger()

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Validate model directory exists and contains expected files
def validate_model_directory(model_dir):
    """Validate that the model directory contains expected model files."""
    if not os.path.exists(model_dir):
        raise RuntimeError(f"Model directory {model_dir} does not exist")
    
    expected_files = ['kmeans_clusterer.joblib']
    for i in range(3):  # Expect cluster models 0, 1, 2
        expected_files.append(f'model_cluster_{i}_DecisionTree.joblib')
        expected_files.append(f'model_cluster_{i}_RandomForest.joblib')
        expected_files.append(f'model_cluster_{i}_GradientBoosting.joblib')
    
    missing_files = []
    for file in expected_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"Missing expected model files: {missing_files}")
    
    return len(missing_files) == 0

# Now you can call the function
deleted_uploads = cleanup_old_files(UPLOAD_FOLDER, 20)
deleted_results = cleanup_old_files(RESULTS_FOLDER, 20)
if deleted_uploads or deleted_results:
    logger.info(f"Startup cleanup: Removed {deleted_uploads} old uploads and {deleted_results} old results")

predictor = None

try:
    # Validate model directory before loading
    validate_model_directory(MODEL_DIR)
    predictor = ModelPredictor(model_dir=MODEL_DIR, logger=logger)
    logger.info("Model predictor initialized successfully")
except Exception as e:
    logger.error(f"Error initializing model predictor: {e}")

# Load environment variables
load_dotenv()
FILENAME_PATTERN = os.getenv('FILENAME_PATTERN', r'^wafer_\d{8}_\d{6}\.csv$')

def is_safe_filename(filename):
    """
    Validate filename to prevent path traversal attacks.
    Returns True if filename is safe, False otherwise.
    """
    if not filename or not isinstance(filename, str):
        return False
    
    # Prevent any directory traversal
    if '/' in filename or '\\' in filename or '..' in filename:
        return False
        
    # Only allow .csv files in results
    if not filename.lower().endswith('.csv'):
        return False
    
    # Additional security checks
    # Prevent null bytes and other dangerous characters
    if '\x00' in filename or any(char in filename for char in ['<', '>', ':', '"', '|', '?', '*']):
        return False
        
    # Limit filename length
    if len(filename) > 255:
        return False
        
    # Only allow alphanumeric, dots, underscores, and hyphens
    import re
    if not re.match(r'^[a-zA-Z0-9._-]+\.csv$', filename):
        return False
        
    return True

@app.route('/')
def index():
    """Main dashboard page"""
    # Get summary of previous predictions if available
    recent_results = None
    prediction_stats = {
        'total': 0,
        'good': 0,
        'bad': 0,
        'percentage_bad': 0
    }
    
    try:
        result_files = sorted(Path(RESULTS_FOLDER).glob("*.csv"), key=os.path.getmtime, reverse=True)
        if result_files:
            recent_file = result_files[0]
            recent_results = pd.read_csv(recent_file)
            
            # Calculate stats
            prediction_stats['total'] = len(recent_results)
            prediction_stats['good'] = len(recent_results[recent_results['status'] == 'Good'])
            prediction_stats['bad'] = len(recent_results[recent_results['status'] == 'Bad'])
            prediction_stats['percentage_bad'] = round((prediction_stats['bad'] / prediction_stats['total']) * 100, 2)
            
            # Get cluster distribution as data
            cluster_counts = recent_results['cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            cluster_data = cluster_counts.to_dict('records')
            
            return render_template('index.html', 
                                   stats=prediction_stats, 
                                   cluster_data=cluster_data,
                                   recent_file=recent_file.name,
                                   has_data=True)
    except Exception as e:
        logger.error(f"Error loading previous results: {e}")
        flash(f"Error loading previous results: {e}", "error")
    
    # If no previous results or error
    return render_template('index.html', stats=prediction_stats, has_data=False)

@app.route('/upload', methods=['GET', 'POST'])
@monitor_performance("File Upload and Processing")
def upload():
    """Upload page for new prediction data"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        # Get list of files (multiple files possible now)
        files = request.files.getlist('file')
        
        if not files or files[0].filename == '':
            flash('No selected files', 'error')
            return redirect(request.url)
        
        successful_uploads = 0
        failed_uploads = 0
        results_filenames = []
        
        for file in files:
            # Validate filename (case-insensitive)
            if not re.match(FILENAME_PATTERN, file.filename, re.IGNORECASE):
                flash(f'Invalid filename: {file.filename}. Must follow pattern: wafer_YYYYMMDD_HHMMSS.csv (case-insensitive)', 'error')
                failed_uploads += 1
                continue
            
            # Save the file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"uploaded_{timestamp}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            try:
                # Process the file
                df = pd.read_csv(filepath)
                logger.info(f"🔍 File uploaded: {filename}, shape: {df.shape}")
                
                # STRICT VALIDATION: Must have exactly 591 columns (Wafer + 590 features)
                EXPECTED_COLUMNS = 591
                if df.shape[1] != EXPECTED_COLUMNS:
                    error_msg = f"❌ FRONTEND REJECTION: Invalid file structure: Expected {EXPECTED_COLUMNS} columns but got {df.shape[1]} columns. File: {file.filename}"
                    logger.error(error_msg)
                    flash(error_msg, 'error')
                    failed_uploads += 1
                    continue
                
                logger.info(f"✅ Frontend column count validation passed: {df.shape[1]} columns")
                logger.info(f"🔍 Original columns (first 5): {list(df.columns[:5])}")
                logger.info(f"🔍 Original columns (last 5): {list(df.columns[-5:])}")
                
                # Standardize column names (except first column which is wafer ID)
                def standardize_column_name(col):
                    """Comprehensive column name standardization."""
                    col = str(col).lower()
                    col = col.replace('-', '_')
                    col = col.replace(' ', '_')
                    col = col.replace('.', '_')
                    col = col.replace('(', '').replace(')', '')
                    col = col.replace('[', '').replace(']', '')
                    # Remove any other non-alphanumeric characters except underscores
                    import re
                    col = re.sub(r'[^\w]', '_', col)
                    # Remove multiple consecutive underscores
                    col = re.sub(r'_+', '_', col)
                    # Remove leading/trailing underscores
                    col = col.strip('_')
                    return col
                
                # Keep first column as is (wafer ID), standardize the rest
                original_column_count = len(df.columns)
                if len(df.columns) > 1:
                    new_columns = [df.columns[0]]  # Keep first column as is
                    new_columns.extend([standardize_column_name(col) for col in df.columns[1:]])
                    df.columns = new_columns
                    logger.info(f"🔍 Standardized columns (first 5): {list(df.columns[:5])}")
                    logger.info(f"🔍 Standardized columns (last 5): {list(df.columns[-5:])}")
                
                # FINAL VALIDATION: Ensure we still have the right number of columns after processing
                if df.shape[1] != EXPECTED_COLUMNS:
                    error_msg = f"❌ FRONTEND PROCESSING ERROR: Started with {original_column_count} columns but ended with {df.shape[1]} columns after standardization"
                    logger.error(error_msg)
                    flash(error_msg, 'error')
                    failed_uploads += 1
                    continue
                
                logger.info(f"✅ Frontend processing complete: {df.shape[1]} columns maintained")
                
                # Run prediction
                if predictor:
                    logger.info(f"🚀 Sending to backend predictor: shape {df.shape}")
                    results = predictor.predict(df)
                    logger.info(f"✅ Backend prediction successful: {len(results)} predictions")
                    
                    # Save results
                    results_filename = f"prediction_results_{timestamp}_{file.filename}"
                    results_path = os.path.join(RESULTS_FOLDER, results_filename)
                    results.to_csv(results_path, index=False)

                    # Clean up old files after new ones are created
                    cleanup_old_files(UPLOAD_FOLDER, 20)
                    cleanup_old_files(RESULTS_FOLDER, 20)

                    results_filenames.append(results_filename)
                    successful_uploads += 1
                else:
                    flash("Model predictor not available", "error")
                    failed_uploads += 1
                    
            except Exception as e:
                logger.error(f"❌ ERROR processing file {file.filename}: {e}")
                logger.error(f"🔍 Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
                flash(f"Error processing file {file.filename}: {e}", "error")
                failed_uploads += 1
        
        # Summary message
        if successful_uploads > 0:
            flash(f"Successfully processed {successful_uploads} file(s)", "success")
            
            # If only one file was uploaded successfully, go to its results page
            if successful_uploads == 1 and len(results_filenames) == 1:
                return redirect(url_for('results', filename=results_filenames[0]))
            else:
                # If multiple files were uploaded, go to results list
                return redirect(url_for('results'))
        else:
            flash("No files were successfully processed", "error")
            
        return redirect(url_for('upload'))
    
    return render_template('upload.html')

@app.route('/results')
def results():
    """Display prediction results."""
    filename = request.args.get('filename')
    
    if filename:
        # Validate filename to prevent path traversal
        if not is_safe_filename(filename):
            flash("Invalid filename", "error")
            logger.warning(f"Security: Blocked access to invalid filename: {filename}")
            return redirect(url_for('results'))
            
        try:
            # Load specific result file
            results_path = os.path.join(RESULTS_FOLDER, filename)
            if not os.path.exists(results_path):
                flash(f"Results file not found: {filename}", "error")
                return redirect(url_for('results'))
                
            results_df = pd.read_csv(results_path)
            logger.info(f"Loaded results file: {filename}, shape: {results_df.shape}")
            
            # Create result_info dict WITHOUT processed_at from DataFrame
            result_info = {
                'filename': filename,
                'total_wafers': len(results_df),
                'good_wafers': len(results_df[results_df['status'] == 'Good']),
                'bad_wafers': len(results_df[results_df['status'] == 'Bad']),
                'percentage_bad': 0
            }
            
            # Use file modification time as processed_at
            file_mtime = os.path.getmtime(results_path)
            result_info['processed_at'] = datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            if result_info['total_wafers'] > 0:
                result_info['percentage_bad'] = round((result_info['bad_wafers'] / result_info['total_wafers']) * 100, 2)
            
            # Create distribution plot
            fig = px.histogram(
                results_df, 
                x='prediction_proba', 
                color='status', 
                marginal='box',
                title='Prediction Probability Distribution'
            )
            prediction_plot = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Create cluster distribution
            cluster_counts = results_df['cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            cluster_fig = px.pie(
                cluster_counts, 
                values='Count', 
                names='Cluster', 
                title='Distribution by Cluster'
            )
            cluster_plot = json.dumps(cluster_fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Convert worst_wafers to a list of dicts for the template
            worst_wafers = results_df.nlargest(10, 'prediction_proba').to_dict('records')
            
            return render_template(
                'results.html', 
                results=results_df.to_dict('records'),
                worst_wafers=worst_wafers,
                result_info=result_info,
                prediction_plot=prediction_plot,
                cluster_plot=cluster_plot,
                filename=filename
            )
            
        except Exception as e:
            logger.error(f"Error loading results file {filename}: {str(e)}")
            flash(f"Error loading results file: {str(e)}", "error")
            return redirect(url_for('results'))
    
    # If no filename or error, show list of all result files
    try:
        result_files = sorted([f.name for f in Path(RESULTS_FOLDER).glob("*.csv")], reverse=True)
        return render_template('results_list.html', result_files=result_files)
    except Exception as e:
        logger.error(f"Error loading results list: {e}")
        flash(f"Error loading results list: {e}", "error")
        return render_template('results_list.html', result_files=[])

@app.route('/api/results/<filename>')
def api_results(filename):
    """API endpoint to get result data as JSON"""
    # Validate filename to prevent path traversal
    if not is_safe_filename(filename):
        return jsonify({
            'status': 'error',
            'message': 'Invalid filename'
        }), 400
        
    try:
        results_path = os.path.join(RESULTS_FOLDER, filename)
        results_df = pd.read_csv(results_path)
        return jsonify({
            'status': 'success',
            'data': results_df.to_dict('records')
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Load debug mode, host, and port from environment variables
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', '5001'))
    
    # Security check: warn if debug mode is enabled
    if debug_mode:
        logger.warning("⚠️  SECURITY WARNING: Debug mode is enabled. This should be disabled in production.")
    
    app.run(debug=debug_mode, host=host, port=port)