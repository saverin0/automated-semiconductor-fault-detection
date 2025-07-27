#!/usr/bin/env python3
"""
Simple test script to validate Flask app routes and templates
"""
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_flask_imports():
    """Test that all Flask dependencies can be imported"""
    try:
        from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
        import pandas as pd
        import numpy as np
        import plotly
        import plotly.express as px
        print("✅ All Flask dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_templates_exist():
    """Test that all required templates exist"""
    templates_dir = project_root / "frontend" / "templates"
    required_templates = [
        "index.html",
        "upload.html", 
        "results.html",
        "results_list.html"
    ]
    
    missing_templates = []
    for template in required_templates:
        template_path = templates_dir / template
        if not template_path.exists():
            missing_templates.append(template)
    
    if missing_templates:
        print(f"❌ Missing templates: {missing_templates}")
        return False
    else:
        print("✅ All required templates exist")
        return True

def test_directory_structure():
    """Test that required directories exist"""
    required_dirs = [
        "uploads",
        "prediction_results", 
        "training_model",
        "frontend/templates"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories exist")
        return True

def main():
    """Run all tests"""
    print("🧪 Testing Flask App Configuration...")
    print("=" * 50)
    
    tests = [
        test_flask_imports,
        test_templates_exist,
        test_directory_structure
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 All tests passed! Flask app should work correctly.")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main()
