# Automated Semiconductor Fault Detection

This project aims to automate the detection of faults in semiconductor manufacturing using machine learning. It provides a robust pipeline for data validation, preprocessing, model training, evaluation, and deployment, ensuring high-quality and reliable fault detection for semiconductor production lines.

## Features

- **Automated Data Validation:** Ensures incoming data meets schema and quality requirements before processing.
- **Data Preprocessing:** Handles missing values, outliers, and feature engineering to prepare data for modeling.
- **Modular Codebase:** Organized into clear modules for validation, transformation, database operations, and modeling.

## Project Structure

```
automated-semiconductor-fault-detection/
├── data/         # Raw and processed datasets
├── logs/         # Logs for pipeline execution and debugging
├── schema/       # Data schema definitions for validation
├── src/          # Source code for validation, preprocessing, modeling, etc.
├── .env          # Environment variable definitions (NEVER commit to version control)
├── LICENSE       # License information (GPL-3.0)
├── README.md     # Project documentation
```

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
  - Remove hard-coded service account emails from code (e.g. `service-account@your-project-id.iam.gserviceaccount.com`)
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
  - Validate model file integrity before loading (use hashes)
- **Path Traversal Prevention:** Always validate filenames:
  - Use `werkzeug.utils.secure_filename()` for uploads
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

## Getting Started

### Prerequisites
- Python 3.7+
- pip (Python package manager)
- (Optional) Virtual environment tool (e.g., `venv` or `conda`)

### Environment Variables

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

### Security Warnings

- Never use hard-coded secrets in production. Always use environment variables.
- Ensure `GOOGLE_APPLICATION_CREDENTIALS` points to a secure service account file.
- **Deserialization Risk:** Only load model files (`joblib.load`) from trusted sources. Malicious files can execute arbitrary code.
- Disable Flask debug mode in production (`FLASK_DEBUG=False`).
