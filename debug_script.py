import re

# Read the current predictor file
with open('src/prediction/predictor.py', 'r') as f:
    content = f.read()

# Add debug logging to preprocess_prediction_data method
preprocess_pattern = r'def preprocess_prediction_data\(self, data: pd\.DataFrame\) -> pd\.DataFrame:.*?return data(\n\s+)except'
preprocess_replacement = r'''def preprocess_prediction_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess prediction data (similar to training preprocessing).
        
        Args:
            data: Input prediction data
            
        Returns:
            Preprocessed data ready for prediction
        """
        try:
            # Debug: Log initial data
            self.logger.info(f"PREPROCESSING - Initial data shape: {data.shape}")
            self.logger.info(f"PREPROCESSING - Initial columns: {data.columns.tolist()}")
            
            # Handle wafer ID column - check multiple possible column names
            wafer_id_columns = ['Wafer', 'Unnamed: 0', 'wafer_id', 'id']
            wafer_ids = None
            
            for col in wafer_id_columns:
                if col in data.columns:
                    wafer_ids = data[col].copy()
                    data = data.drop(col, axis=1)
                    self.logger.info(f"Found wafer ID column: {col}")
                    break
            
            if wafer_ids is None:
                wafer_ids = data.index
                self.logger.info("Using row index as wafer IDs")
            
            # Handle missing values
            data = data.replace('?', np.nan)
            
            # Remove columns with all null values
            data = data.dropna(axis=1, how='all')
            self.logger.info(f"PREPROCESSING - After dropping null columns: {data.shape}")
            
            # Fill remaining missing values with mean (for numeric) or mode (for categorical)
            for column in data.columns:
                if data[column].dtype in ['float64', 'int64']:
                    data[column].fillna(data[column].mean(), inplace=True)
                else:
                    data[column].fillna(data[column].mode()[0] if not data[column].mode().empty else 'Unknown', inplace=True)
            
            # Debug: Log data before numeric conversion
            self.logger.info(f"PREPROCESSING - Before numeric conversion: {data.shape}")
            
            # Convert all columns to numeric where possible, but be more careful
            numeric_features = []
            for column in data.columns:
                try:
                    # Only convert if all non-null values can be converted to numeric
                    pd.to_numeric(data[column], errors='raise')
                    data[column] = pd.to_numeric(data[column], errors='coerce')
                    numeric_features.append(column)
                except (ValueError, TypeError):
                    # If conversion fails, keep as-is and warn
                    self.logger.warning(f"Column {column} contains non-numeric data, keeping as-is")
            
            # Debug: Log numeric features
            self.logger.info(f"PREPROCESSING - Numeric features: {len(numeric_features)}/{len(data.columns)}")
            
            # Store wafer IDs for later use
            data['wafer_id'] = wafer_ids
            
            # Debug: Log final data
            self.logger.info(f"PREPROCESSING - Final data shape: {data.shape}")
            self.logger.info(f"PREPROCESSING - Final columns: {data.columns.tolist()}")
            
            return data
            
        except\1'''

# Add debug logging to predict method
predict_pattern = r'def predict\(self, input_file_path: str\) -> str:.*?try:.*?data = pd\.read_csv\(input_file_path\)'
predict_replacement = r'''def predict(self, input_file_path: str) -> str:
        """
        Make predictions on input CSV file.
        
        Args:
            input_file_path: Path to input CSV file with wafer data
            
        Returns:
            Path to saved predictions CSV file
        """
        try:
            # Debug: Log start of prediction
            self.logger.info(f"Starting prediction on file: {input_file_path}")
            
            # Load data
            data = pd.read_csv(input_file_path)
            self.logger.info(f"PREDICTION - Input data shape: {data.shape}")
            self.logger.info(f"PREDICTION - Input columns: {data.columns.tolist()}")'''

model_predict_pattern = r'# Apply model to make predictions.*?if self\.models:'
model_predict_replacement = r'''# Apply model to make predictions
            if self.models:
                # Debug: Check model features
                model = list(self.models.values())[0]
                if hasattr(model, 'feature_names_in_'):
                    model_features = model.feature_names_in_.tolist()
                    self.logger.info(f"MODEL - Expected features: {model_features}")
                    data_features = processed_data.columns.tolist()
                    if 'wafer_id' in data_features:
                        data_features.remove('wafer_id')
                    self.logger.info(f"DATA - Available features: {data_features}")
                    
                    # Check for missing features
                    missing_features = set(model_features) - set(data_features)
                    extra_features = set(data_features) - set(model_features)
                    
                    if missing_features:
                        self.logger.warning(f"MISSING FEATURES: {missing_features}")
                    if extra_features:
                        self.logger.info(f"EXTRA FEATURES: {extra_features}")'''

# Apply all replacements
content = re.sub(preprocess_pattern, preprocess_replacement, content, flags=re.DOTALL)
content = re.sub(predict_pattern, predict_replacement, content, flags=re.DOTALL)
content = re.sub(model_predict_pattern, model_predict_replacement, content, flags=re.DOTALL)

# Write the modified file
with open('src/prediction/predictor.py', 'w') as f:
    f.write(content)

print("Added debug logging to the predictor code!")
