# Semiconductor Wafer Fault Detection - Project Overview

## 🎯 Problem Statement

The inputs of various sensors for different wafers have been provided. In electronics, a **wafer** (also called a slice or substrate) is a thin slice of semiconductor used for the fabrication of integrated circuits. 

### **Project Goal**
Build a machine learning model which predicts whether a wafer needs to be replaced or not (i.e., whether it is working or not) based on the inputs from various sensors.

### **Classification Classes**
There are two classes: **+1** and **-1**

- **+1** means that the wafer is in a **working condition** and it **doesn't need to be replaced**
- **-1** means that the wafer is **faulty** and it **needs to be replaced**

## 🏗️ Architecture Diagrams

### 2.1 Functional Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    WAFER FAULT DETECTION SYSTEM                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   INPUT     │    │  VALIDATION │    │  PROCESSING │         │
│  │             │    │             │    │             │         │
│  │ • Wafer     │───▶│ • Schema    │───▶│ • Data      │         │
│  │   Sensor    │    │   Validation│    │   Preprocess│         │
│  │   Data      │    │ • Good/Bad  │    │ • Clustering│         │
│  │ • Schema    │    │   Separation│    │ • Model     │         │
│  │   Files     │    │ • Error     │    │   Training  │         │
│  │             │    │   Reporting │    │ • Prediction│         │
│  │             │    │             │    │             │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   OUTPUT    │    │   DATABASE  │    │  DEPLOYMENT │         │
│  │             │    │             │    │             │         │
│  │ • Fault     │    │ • BigQuery  │    │ • Cloud     │         │
│  │   Detection │    │ • Data      │    │   Platform  │         │
│  │   Results   │    │   Storage   │    │ • Web       │         │
│  │ • Good/Bad  │    │ • Export    │    │   Interface │         │
│  │   Status    │    │   Pipeline  │    │ • API       │         │
│  │ • Confidence│    │             │    │   Endpoints │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Technical Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                        TECHNICAL STACK                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   FRONTEND      │  │   BACKEND       │  │   DATABASE      │  │
│  │                 │  │                 │  │                 │  │
│  │ • HTML5         │  │ • Python 3.8+   │  │ • Google Cloud  │  │
│  │ • CSS3          │  │ • Flask         │  │   BigQuery      │  │
│  │ • JavaScript    │  │ • Pandas        │  │ • SQL           │  │
│  │ • Bootstrap     │  │ • NumPy         │  │                 │  │
│  │ • Chart.js      │  │ • Scikit-learn  │  │                 │  │
│  └─────────────────┘  │ • XGBoost       │  └─────────────────┘  │
│                       │ • Joblib        │                       │
│                       └─────────────────┘                       │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   MACHINE       │  │   DEPLOYMENT    │  │   VALIDATION    │  │
│  │   LEARNING      │  │                 │  │                 │  │
│  │                 │  │                 │  │                 │  │
│  │ • K-Means       │  │ • Pivotal Cloud │  │ • JSON Schema   │  │
│  │   Clustering    │  │   Foundry       │  │ • Regex         │  │
│  │ • Random Forest │  │ • Docker        │  │   Validation    │  │
│  │ • XGBoost       │  │ • Git           │  │ • Data Type     │  │
│  │ • GridSearch    │  │ • CI/CD         │  │   Validation    │  │
│  │ • KNN Imputer   │  │                 │  │ • Null Value    │  │
│  └─────────────────┘  └─────────────────┘  │   Check         │  │
│                                            └─────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   CLIENT    │    │   VALIDATION│    │   DATABASE  │    │   ML MODEL  │
│   DATA      │───▶│   LAYER     │───▶│   LAYER     │───▶│   LAYER     │
│             │    │             │    │             │    │             │
│ • CSV Files │    │ • Schema    │    │ • BigQuery  │    │ • Clustering│
│ • Schema    │    │   Validation│    │ • Tables    │    │ • Training  │
│   Files     │    │ • Data Type │    │ • Data      │    │ • Prediction│
│             │    │ • Null Check│    │   Export    │    │             │
│             │    │ • Good/Bad  │    │             │    │             │
│             │    │   Separation│    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   WEB       │    │   LOGGING   │    │   CLOUD     │    │   RESULTS   │
│   INTERFACE │    │   SYSTEM    │    │   STORAGE   │    │   EXPORT    │
│             │    │             │    │             │    │             │
│ • Upload    │    │ • Error     │    │ • Model     │    │ • CSV       │
│   Interface │    │   Logs      │    │   Files     │    │   Reports   │
│ • Results   │    │ • Process   │    │ • Data      │    │ • API       │
│   Display   │    │   Logs      │    │   Backups   │    │   Response  │
│ • API       │    │ • Audit     │    │             │    │             │
│   Endpoints │    │   Trails    │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## 📊 System Components

### **Data Processing Pipeline**
1. **Data Ingestion**: CSV files with 590 sensor columns + wafer ID
2. **Data Validation**: Schema-based validation with Good/Bad file separation
3. **Database Upload**: Only validated "good" files are uploaded to database
4. **Data Export**: Validated data exported from database for processing
5. **Preprocessing**: KNN imputation, feature selection, normalization
6. **Clustering**: K-Means clustering for pattern identification
7. **Modeling**: Multi-algorithm approach (Random Forest, XGBoost)
8. **Prediction**: Real-time fault detection with confidence scores

### **Key Features**
- **Automated Validation**: Comprehensive data quality checks with Good/Bad file separation
- **Scalable Architecture**: Cloud-based deployment
- **Real-time Processing**: Instant fault detection
- **Multi-model Approach**: Cluster-specific model selection
- **Web Interface**: User-friendly upload and results display
- **API Integration**: RESTful endpoints for programmatic access

### **Data Validation Process**
The system implements a robust validation pipeline that occurs **FIRST** in the workflow:

1. **Input Validation**: Files are checked for proper naming conventions and format
2. **Schema Validation**: Column structure and data types are validated against predefined schemas
3. **Content Validation**: Data quality checks including null value detection
4. **File Separation**: Valid files are moved to "good" directory, invalid files to "bad" directory
5. **Error Reporting**: Detailed error logs are generated for rejected files
6. **Pipeline Continuation**: Only "good" files proceed to database upload and subsequent processing

This validation-first approach ensures data quality and prevents downstream processing errors.

### **Performance Metrics**
- **Accuracy**: High prediction accuracy for fault detection
- **Speed**: Fast processing of batch data
- **Reliability**: Robust error handling and validation
- **Scalability**: Cloud deployment for production use

## 🎯 Business Impact

### **Manufacturing Benefits**
- **Early Fault Detection**: Identify defective wafers before final testing
- **Cost Reduction**: Prevent defective chips from reaching customers
- **Quality Improvement**: Maintain high manufacturing standards
- **Process Optimization**: Data-driven insights for process improvement

### **Operational Benefits**
- **Automation**: Reduce manual inspection requirements
- **Efficiency**: Faster quality control processes
- **Consistency**: Standardized fault detection across batches
- **Traceability**: Complete audit trail of predictions and decisions

## 🔧 Implementation Phases

### **Phase 1: Development & Testing**
- Data pipeline development
- Model training and validation
- Local testing and optimization

### **Phase 2: Deployment**
- Cloud platform setup
- Application deployment
- Integration testing

### **Phase 3: Production**
- Live system monitoring
- Performance optimization
- Continuous improvement

This overview provides a comprehensive understanding of the wafer fault detection system, its architecture, and technical implementation details.
