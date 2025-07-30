# Automated Semiconductor Fault Detection

This project aims to automate the detection of faults in semiconductor manufacturing using machine learning. It provides a robust pipeline for data validation, preprocessing, model training, evaluation, and deployment, ensuring high-quality and reliable fault detection for semiconductor production lines.

## Features

- **Automated Data Validation:** Ensures incoming data meets schema and quality requirements before processing
- **Data Preprocessing:** Handles missing values, outliers, and feature engineering to prepare data for modeling
- **Modular Codebase:** Organized into clear modules for validation, transformation, database operations, and modeling
- **Web Interface:** Flask-based web application for easy data upload and result visualization
- **Performance Monitoring:** Built-in performance tracking and memory usage monitoring
- **Security Hardened:** Comprehensive security measures including input validation and model file integrity checks
- **Type Safety:** Full type hints for better code quality and IDE support

## Project Structure

```
automated-semiconductor-fault-detection/
├── data/                    # Raw and processed datasets
├── frontend/               # Flask web application
│   ├── app.py             # Main Flask application
│   └── templates/         # HTML templates
├── logs/                   # Logs for pipeline execution and debugging
├── prediction_results/      # Prediction output files
├── schema/                 # Data schema definitions for validation
├── src/                    # Source code modules
│   ├── best_model_finder/  # Model selection and tuning
│   ├── data_ingestion/     # Database operations
│   ├── data_preprocessing/ # Data cleaning and feature engineering
│   ├── data_validation/    # Data quality checks
│   ├── model_testing/      # Prediction functionality
│   └── utils/              # Configuration and utilities
├── training_model/         # Trained ML models
├── uploads/                # Uploaded prediction files
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
   Create a `.env` file in the project root with the following keys:

   ```
   FLASK_SECRET_KEY=your-secure-secret-key
   FLASK_DEBUG=False
   FLASK_HOST=0.0.0.0
   FLASK_PORT=5001
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
   BIGQUERY_SERVICE_ACCOUNT=your-service-account@project-id.iam.gserviceaccount.com
   MODEL_SAVE_DIR=training_model
   UPLOAD_FOLDER=uploads
   RESULTS_DIR=prediction_results
   ```

   **Important:** `FLASK_SECRET_KEY` is required. If it is missing, the web application will abort during startup.

4. **Run the web application:**
   ```bash
   python frontend/app.py
   ```

5. **Access the web interface:**
   Open your browser and navigate to `http://localhost:5001`

### Usage

#### Web Interface
- **Upload Data:** Use the web interface to upload CSV files for prediction
- **View Results:** See prediction results with interactive visualizations
- **Download Results:** Export prediction results as CSV files

#### Command Line
- **Training Pipeline:** `python main_pipeline.py --mode training`
- **Prediction Pipeline:** `python main_pipeline.py --mode prediction`
- **Full Pipeline:** `python main_pipeline.py --mode full`

### Security Warnings

- Never use hard-coded secrets in production. Always use environment variables.
- Ensure `GOOGLE_APPLICATION_CREDENTIALS` points to a secure service account file.
- **Deserialization Risk:** Only load model files (`joblib.load`) from trusted sources. Malicious files can execute arbitrary code.
- Disable Flask debug mode in production (`FLASK_DEBUG=False`).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.
