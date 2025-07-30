# Automated Semiconductor Wafer Fault Detection

## 🏭 What is Semiconductor Wafer Fault Detection?

This project implements an **automated machine learning system for detecting manufacturing faults in semiconductor wafers**. Semiconductor wafers are thin slices of silicon used to fabricate integrated circuits (chips). During the manufacturing process, various defects can occur that affect the quality and yield of the final products.

### 🎯 Project Purpose

**Semiconductor manufacturing** is a highly complex process where even minor defects can lead to significant yield losses. This system helps:

- **Detect manufacturing faults** in real-time during wafer production
- **Improve yield rates** by identifying defective wafers early
- **Reduce costs** by preventing defective chips from reaching final testing
- **Ensure quality** in semiconductor manufacturing processes
- **Automate quality control** to reduce human error and increase efficiency

### 🔬 What are Semiconductor Wafers?

**Semiconductor wafers** are the foundation of modern electronics:
- **Material**: Primarily silicon, sometimes other semiconductors
- **Size**: Typically 150mm, 200mm, or 300mm in diameter
- **Process**: Undergo multiple manufacturing steps (lithography, etching, doping, etc.)
- **Sensors**: Each wafer has hundreds of sensors monitoring various parameters
- **Data**: Generates massive amounts of sensor data during manufacturing

### 🚨 Why Fault Detection is Critical

**Manufacturing defects** can occur at any stage:
- **Chemical contamination** from processing steps
- **Physical damage** from handling or equipment
- **Electrical faults** from improper doping or metallization
- **Pattern defects** from lithography issues
- **Environmental factors** like temperature, humidity, or particle contamination

**Early detection** prevents:
- ❌ **Defective chips** reaching customers
- ❌ **Wasted manufacturing resources**
- ❌ **Production line delays**
- ❌ **Quality control failures**

## 🚀 Features

- **Automated Data Validation:** Ensures incoming wafer sensor data meets schema and quality requirements before processing
- **Data Preprocessing:** Handles missing values, outliers, and feature engineering to prepare wafer data for modeling
- **Modular Codebase:** Organized into clear modules for validation, transformation, database operations, and modeling
- **Web Interface:** Flask-based web application for easy wafer data upload and fault detection results visualization
- **Performance Monitoring:** Built-in performance tracking and memory usage monitoring for production environments
- **Security Hardened:** Comprehensive security measures including input validation and model file integrity checks
- **Type Safety:** Full type hints for better code quality and IDE support
- **Real-time Predictions:** Instant fault detection with detailed confidence scores and cluster assignments

## 📊 Data Structure

Each wafer file contains:
- **Wafer ID**: Unique identifier for each wafer
- **590 Sensor Readings**: Various manufacturing parameters (temperature, pressure, chemical concentrations, etc.)
- **Expected Format**: CSV files with 591 columns (wafer_id + 590 sensors)
- **File Naming**: `wafer_YYYYMMDD_HHMMSS.csv` (case-insensitive)

## 🏗️ Project Structure

```
automated-semiconductor-fault-detection/
├── data/                    # Raw and processed wafer datasets
│   ├── training/           # Training data for model development
│   └── prediction/         # New wafer data for fault detection
├── frontend/               # Flask web application
│   ├── app.py             # Main Flask application
│   └── templates/         # HTML templates for web interface
├── logs/                   # Logs for pipeline execution and debugging
├── prediction_results/      # Fault detection output files
├── schema/                 # Data schema definitions for validation
├── src/                    # Source code modules
│   ├── best_model_finder/  # Model selection and tuning
│   ├── data_ingestion/     # Database operations
│   ├── data_preprocessing/ # Data cleaning and feature engineering
│   ├── data_validation/    # Data quality checks
│   ├── model_testing/      # Prediction functionality
│   └── utils/              # Configuration and utilities
├── training_model/         # Trained ML models for fault detection
├── uploads/                # Uploaded wafer files
├── .env                    # Environment variables (NEVER commit)
├── .gitignore             # Git ignore rules
├── LICENSE                 # License information
├── README.md              # This documentation
├── requirements.txt        # Python dependencies
└── main_pipeline.py       # Main orchestration pipeline
```

## Recent Improvements

### ✅ Security Enhancements
- **Removed hardcoded secrets** - All sensitive data now uses environment variables
- **Enhanced input validation** - Comprehensive filename and path validation
- **Model file integrity checks** - Validates model files before loading
- **Debug mode warnings** - Alerts when debug mode is enabled in production
- **Path traversal protection** - Prevents directory traversal attacks

### ✅ Code Quality Improvements
- **Type hints** - Full type annotations for better IDE support
- **Performance monitoring** - Built-in execution time and memory tracking
- **Centralized configuration** - Unified config management
- **Better error handling** - Specific exception types and detailed logging
- **Code cleanup** - Removed redundant files and unused code

### ✅ New Features
- **Web interface** - User-friendly Flask application for data upload
- **Real-time predictions** - Instant fault detection results
- **Visualization** - Interactive charts and statistics
- **File management** - Automatic cleanup of old files
- **API endpoints** - RESTful API for programmatic access

## Security Best Practices

### Environment Variables
- **Never commit `.env` to version control** - Add it to `.gitignore`
- Generate a secure random Flask secret key:
  ```bash
  python -c "import secrets; print(secrets.token_hex(32))"
  ```
- Store all credentials as environment variables, never in code

### GCP Database Security
- **Service Account Impersonation:** The application uses impersonated credentials for BigQuery access
  - Remove hard-coded service account emails from code
  - Store service account identifier in environment variables
  - Keep service account keys with minimum required permissions
  - **Auto-refresh Issue:** Impersonated credentials can expire during long operations
    - Implement retry logic with exponential backoff for database operations
    - Monitor for "Request had invalid authentication credentials" errors
    - Consider using Workload Identity Federation instead of service account keys when possible

### Deployment Security
- **Disable debug mode** in production: `FLASK_DEBUG=False`
- Use a production WSGI server (Gunicorn, uWSGI) instead of Flask's built-in server:
  ```bash
  gunicorn --workers=4 --bind=0.0.0.0:5001 frontend.app:app
  ```
- Set appropriate file permissions:
  ```bash
  chmod 600 .env                     # Only owner can read/write
  chmod -R 700 training_model/       # Secure model directory
  chmod 600 service_account.json     # Protect credentials
  ```

### Code Security
- **Model Deserialization Risk:** Only load model files from trusted sources
  - Malicious joblib/pickle files can execute arbitrary code
  - Validate model file integrity before loading (implemented)
- **Path Traversal Prevention:** Always validate filenames:
  - Comprehensive filename validation implemented
  - Validate file extensions and paths before accessing
- **API Security:** Implement rate limiting and authentication for API endpoints
- Regularly update dependencies to patch security vulnerabilities:
  ```bash
  pip install --upgrade -r requirements.txt
  ```

### Cloud Security
- Use service accounts with minimum required permissions
- Rotate service account keys regularly
- Enable audit logging for all cloud resources

## Security Checklist Before Deployment

Before deploying this application to production, ensure you have completed the following security measures:

### ✅ Environment Variables
- [ ] Set `FLASK_SECRET_KEY` to a secure random string (32+ characters)
- [ ] Set `FLASK_DEBUG=False` in production
- [ ] Configure all Google Cloud credentials via environment variables
- [ ] Never commit `.env` files to version control

### ✅ File Permissions
- [ ] Set restrictive permissions on model files: `chmod 600 training_model/*`
- [ ] Protect upload and results directories: `chmod 700 uploads/ prediction_results/`
- [ ] Secure any service account keys: `chmod 600 service_account.json`

### ✅ Application Security
- [ ] Use HTTPS in production (configure reverse proxy with SSL)
- [ ] Implement rate limiting for API endpoints
- [ ] Add authentication if needed for your use case
- [ ] Regularly update dependencies: `pip install --upgrade -r requirements.txt`

### ✅ Model Security
- [ ] Validate model files before loading (implemented in app.py)
- [ ] Only load models from trusted sources
- [ ] Consider model file integrity checks (SHA256 hashes)

### ✅ Cloud Security
- [ ] Use service accounts with minimum required permissions
- [ ] Enable audit logging for BigQuery operations
- [ ] Rotate service account keys regularly
- [ ] Consider using Workload Identity Federation

## Getting Started

### Prerequisites
- Python 3.8+ (tested with Python 3.13.5)
- pip (Python package manager)
- (Optional) Virtual environment tool (e.g., `venv` or `conda`)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd automated-semiconductor-fault-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   
   **Option A: Use the provided template (Recommended)**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and set your actual values, especially `FLASK_SECRET_KEY`.

   **Option B: Create manually**
   Create a `.env` file in the project root with the following keys:

   ```
   FLASK_SECRET_KEY=your-secure-secret-key
   FLASK_DEBUG=False
   FLASK_HOST=0.0.0.0
   FLASK_PORT=5001
   MODEL_SAVE_DIR=training_model
   UPLOAD_FOLDER=uploads
   RESULTS_DIR=prediction_results
   ```

   **Important:** 
   - `FLASK_SECRET_KEY` is required. If it is missing, the web application will abort during startup.
   - Generate a secure key: `python -c "import secrets; print(secrets.token_hex(32))"`
   - Never commit your `.env` file to version control (it's already in `.gitignore`)

4. **Run the web application:**
   ```bash
   python frontend/app.py
   ```

5. **Access the web interface:**
   Open your browser and navigate to `http://localhost:5001`

### Usage

#### Web Interface
- **Upload Wafer Data:** Use the web interface to upload wafer CSV files for fault detection
- **View Fault Detection Results:** See fault detection results with interactive visualizations showing:
  - **Fault Status**: Good/Bad classification for each wafer
  - **Confidence Scores**: Probability of fault detection
  - **Cluster Assignment**: Which manufacturing pattern the wafer follows
  - **Sensor Analysis**: Key sensor readings that influenced the prediction
- **Download Results:** Export fault detection results as CSV files for further analysis

#### Command Line
- **Training Pipeline:** `python main_pipeline.py --mode training`
  - Trains fault detection models on historical wafer data
  - Performs clustering to identify different manufacturing patterns
  - Saves trained models for real-time fault detection

- **Prediction Pipeline:** `python main_pipeline.py --mode prediction`
  - Processes new wafer data for fault detection
  - Generates fault detection reports
  - Saves results for quality control analysis

#### API Endpoints
- **POST /upload**: Upload wafer CSV files for fault detection
- **GET /results**: View fault detection results
- **GET /results/<filename>**: Download specific fault detection results

### 📋 Example Workflow

1. **Prepare Wafer Data**: Ensure your CSV file has 591 columns (wafer_id + 590 sensors)
2. **Upload via Web Interface**: Navigate to `http://localhost:5001/upload`
3. **Upload Wafer File**: Select your wafer CSV file (format: `wafer_YYYYMMDD_HHMMSS.csv`)
4. **View Results**: See real-time fault detection with confidence scores
5. **Download Report**: Export results for quality control documentation

### 🔍 Understanding Results

**Fault Detection Output:**
- **Status**: "Good" (no faults detected) or "Bad" (faults detected)
- **Confidence**: Probability score (0.0-1.0) indicating prediction certainty
- **Cluster**: Manufacturing pattern classification (0, 1, or 2)
- **Timestamp**: When the fault detection was performed

**Quality Control Actions:**
- **Good Wafers**: Continue to next manufacturing step
- **Bad Wafers**: Flag for inspection, rework, or disposal
- **High Confidence**: Trust the automated decision
- **Low Confidence**: Require manual inspection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.
