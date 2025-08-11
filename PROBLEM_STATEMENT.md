# Problem Statement: Automated Semiconductor Wafer Fault Detection

## 🎯 Project Objective

To build a **classification methodology** to predict the quality of wafer sensors based on the given training data using machine learning techniques.

## 🏗️ System Architecture

Our automated semiconductor fault detection system follows a comprehensive machine learning pipeline with three main phases:

### **Phase I: Model Training & Optimization**
```
Start → Data (Batches) for Training → Data Validation → Data Transformation → 
Data Insertion in Database → Export Data from Database to CSV → Data Preprocessing → 
Data Clustering → Get Best Model of Each Cluster → Hyperparameter Tuning
```

### **Phase II: Model Deployment**
```
Hyperparameter Tuning → Model Saving → Cloud Setup → Pushing App to Cloud → Application Start
```

### **Phase III: Real-time Prediction**
```
Application Start → Data from Client → Data Validation → Data Transformation → 
Data Insertion in Database → Export Data from Database to CSV → Data Preprocessing → 
Data Clustering → Model Call for Specific Cluster → Prediction → Export Prediction to CSV → End
```

## 📊 Data Description

### Training Data Structure
The client will send data in multiple sets of files in batches at a given location. Data will contain:

- **Wafer Names**: Unique identifiers for each wafer
- **590 Sensor Columns**: Different sensor values for each wafer
- **Target Column**: "Good/Bad" classification for each wafer

### Target Variable Encoding
- **"+1"** represents **Bad wafer** (faulty)
- **"-1"** represents **Good wafer** (non-faulty)

### Schema Requirements
Apart from training files, we also require a **"schema" file** from the client, which contains all the relevant information about the training files such as:
- Name of the files
- Length of Date value in FileName
- Length of Time value in FileName
- Number of Columns
- Name of the Columns
- Data types of columns

## 🔍 Data Validation Process

In this step, we perform different sets of validation on the given set of training files:

### 1. **Name Validation**
- We validate the name of the files based on the given name in the schema file
- Created a regex pattern as per the name given in the schema file for validation
- Check for the length of date in the file name as well as the length of time in the file name
- **Result**: Valid files → "Good_Data_Folder", Invalid files → "Bad_Data_Folder"

### 2. **Number of Columns Validation**
- Validate the number of columns present in the files
- Must match with the value given in the schema file
- **Result**: Mismatch → "Bad_Data_Folder"

### 3. **Column Names Validation**
- The name of the columns must be the same as given in the schema file
- **Result**: Mismatch → "Bad_Data_Folder"

### 4. **Data Type Validation**
- The datatype of columns is validated when inserting files into Database
- **Result**: Wrong datatype → "Bad_Data_Folder"

### 5. **Null Values Validation**
- If any of the columns in a file have all the values as NULL or missing, we discard such a file
- **Result**: All null columns → "Bad_Data_Folder"

## 🗄️ Data Insertion in Database

### 1. **Database Creation and Connection**
- Create a database with the given name passed
- If the database is already created, open the connection to the database

### 2. **Table Creation**
- Table with name "Good_Data" is created in the database
- Based on given column names and datatype in the schema file
- If the table is already present, new files are inserted in the existing table
- **Purpose**: Training to be done on new as well as old training files

### 3. **File Insertion**
- All files in the "Good_Data_Folder" are inserted in the above-created table
- If any file has invalid data type in any of the columns, the file is not loaded
- **Result**: Invalid files → "Bad_Data_Folder"

## 🤖 Model Training Process

### 1. **Data Export from Database**
- The data stored in the database is exported as a CSV file for model training

### 2. **Data Preprocessing**
   - **Null Value Handling**: Check for null values in the columns. If present, impute using KNN imputer
   - **Feature Selection**: Check if any column has zero standard deviation, remove such columns as they don't provide information during model training

### 3. **Clustering**
- **Algorithm**: KMeans algorithm is used to create clusters in the preprocessed data
- **Optimum Clusters**: Selected by plotting the elbow plot
- **Dynamic Selection**: Using "KneeLocator" function for automatic cluster number selection
- **Purpose**: Implement different algorithms to train data in different clusters
- **Output**: KMeans model is trained and saved for further use in prediction

### 4. **Model Selection**
- **Algorithms Used**: "Random Forest" and "XGBoost"
- **Process**: For each cluster, both algorithms are passed with best parameters derived from GridSearch
- **Selection Criteria**: Calculate AUC scores for both models and select the model with the best score
- **Output**: Best model for each cluster is saved for use in prediction

## 📈 Prediction Data Description

### Input Data Structure
Client will send the data in multiple sets of files in batches at a given location. Data will contain:
- **Wafer Names**: Unique identifiers for each wafer
- **590 Sensor Columns**: Different sensor values for each wafer

### Schema Requirements
Same schema file requirements as training data:
- Name of the files
- Length of Date value in FileName
- Length of Time value in FileName
- Number of Columns
- Name of the Columns
- Data types of columns

## 🔍 Prediction Data Validation

Same validation process as training data:
1. **Name Validation** - Regex pattern validation
2. **Number of Columns** - Column count validation
3. **Name of Columns** - Column name validation
4. **Datatype of Columns** - Data type validation
5. **Null Values** - Missing data validation

## 🗄️ Prediction Data Insertion

Same database insertion process as training:
1. **Database Creation and Connection**
2. **Table Creation** - "Good_Data" table
3. **File Insertion** - Valid files only

## 🎯 Prediction Process

### 1. **Data Export from Database**
- The data in the stored database is exported as a CSV file for prediction

### 2. **Data Preprocessing**
   - **Null Value Handling**: Check for null values, impute using KNN imputer
   - **Feature Selection**: Remove columns with zero standard deviation (same as training)

### 3. **Clustering**
- **Model Loading**: KMeans model created during training is loaded
- **Cluster Prediction**: Clusters for the preprocessed prediction data are predicted

### 4. **Prediction**
- **Model Selection**: Based on the cluster number, the respective model is loaded
- **Prediction**: Model is used to predict the data for that cluster

### 5. **Result Generation**
- Once prediction is made for all clusters, predictions along with Wafer names are saved in a CSV file
- The location is returned to the client

## ☁️ Deployment

### Platform: Pivotal Cloud Foundry (PCF)

### Deployment Files Structure:
```
├── requirements.txt    # All packages needed for cloud deployment
├── main.py            # Entry point of application (Flask server)
├── obj.py             # Prediction logic based on input data
├── manifest.yml       # Instance configuration, app name, build pack language
├── Procfile           # Entry point of the app
└── runtime.txt        # Python version number
```

### Cloud Deployment Process:

1. **Platform Setup**
   - Visit https://pivotal.io/platform
   - Start trial and create account
   - Download CLI for Windows 64-bit
   - Complete email verification

2. **CLI Installation**
   - Unzip CLI file and install .exe with admin rights
   - Verify installation: `cf` command in CMD
   - Login: `cf login -a https://api.run.pivotal.io`

3. **Application Deployment**
   - Navigate to project folder
   - Run: `cf push`
   - Application deployed with route URL

4. **Testing**
   - Use Postman or similar tool to test the deployed application
   - Verify prediction functionality

## 🔄 Workflow Summary

### Training Workflow:
```
Schema File → Data Validation → Database Insertion → Data Export → 
Preprocessing → Clustering → Model Selection → Model Saving
```

### Prediction Workflow:
```
Schema File → Data Validation → Database Insertion → Data Export → 
Preprocessing → Clustering → Model Loading → Prediction → Results Export
```

## 🎯 Key Features

- **Automated Data Validation**: Comprehensive validation pipeline
- **Database Integration**: Secure data storage and retrieval
- **Clustering-based Modeling**: KMeans clustering for pattern identification
- **Multi-algorithm Selection**: Random Forest and XGBoost with GridSearch
- **Cloud Deployment**: Pivotal Cloud Foundry platform
- **Real-time Prediction**: Instant fault detection capabilities
- **Scalable Architecture**: Batch processing support

## 📋 Success Criteria

- **Accuracy**: High prediction accuracy for wafer fault detection
- **Performance**: Fast processing of batch data
- **Reliability**: Robust validation and error handling
- **Scalability**: Cloud deployment for production use
- **Maintainability**: Modular codebase with clear documentation
