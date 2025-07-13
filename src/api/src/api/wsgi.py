"""
WSGI entry point for production deployment
"""
import os
from src.api.app import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)