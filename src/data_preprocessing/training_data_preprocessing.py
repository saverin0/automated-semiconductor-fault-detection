import pandas as pd
import numpy as np
import sys
import os
import logging
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import joblib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import from the project
from src.best_model_finder.tuner import Model_Finder
from src.utils.logging_utils import setup_logger as create_logger

def setup_logger():
    """Set up a logger with both console and file output."""
    log_file = os.getenv('TRAINING_PREPROCESSING_LOG', 'training_preprocessing.log')
    return create_logger('training_preprocessing', log_file)

class KMeansClustering:
    def __init__(self):
        self.clusterer = None  # Add this line to store the KMeans model
        
    def elbow_plot(self, X, logger, max_clusters=10):
        """
        Uses the elbow method to determine the optimal number of clusters.
        Returns the optimal number of clusters.
        """
        logger.info("Starting elbow plot to determine optimal number of clusters.")
        logger.info(f"Input data shape: {X.shape}")
        wcss = []
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
            logger.info(f"Clusters: {i}, WCSS: {kmeans.inertia_:.2f}")
        logger.info(f"Elbow plot WCSS values: {wcss}")
        
        # Implement elbow method: find the point where adding clusters doesn't help much
        if len(wcss) < 3:
            optimal_clusters = min(3, len(wcss))
        else:
            # Calculate the rate of change in WCSS
            wcss_diff = np.diff(wcss)
            wcss_diff_ratio = np.diff(wcss_diff)
            
            # Find the elbow point (where the rate of change starts to level off)
            # Look for the point where the second derivative is minimized
            elbow_idx = np.argmin(wcss_diff_ratio) + 2  # +2 because we took two diffs
            optimal_clusters = max(2, min(elbow_idx, max_clusters))
        
        logger.info(f"Optimal number of clusters selected: {optimal_clusters}")
        logger.info("Elbow plot completed.")
        return optimal_clusters

    def create_clusters(self, X, n_clusters, logger):
        """
        Assigns each sample in X to a cluster.
        Returns a numpy array of cluster labels.
        """
        logger.info(f"Creating {n_clusters} clusters using KMeans.")
        logger.info(f"Input data shape for clustering: {X.shape}")
        
        # Create and save the KMeans model as an instance attribute
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.clusterer.fit_predict(X)
        
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        logger.info(f"Clusters assigned: {unique_clusters}")
        logger.info(f"Cluster distribution: {dict(zip(unique_clusters, counts))}")
        logger.info("Clustering completed.")
        return clusters

class DataTransform:
    def __init__(self, input_file):
        self.input_file = input_file

    def replaceMissingWithNull(self):
        df = pd.read_csv(self.input_file)
        df.replace("?", np.nan, inplace=True)
        df.to_csv(self.input_file, index=False)

class Preprocessor:
    def __init__(self, logger=None):
        self.logger = logger

    def remove_wafer_column(self, df):
        if 'Wafer' in df.columns:
            df = df.drop('Wafer', axis=1)
            if self.logger:
                self.logger.info("Removed 'Wafer' column.")
        return df

    def separate_features_and_label(self, df, label_column_options=None):
        if label_column_options is None:
            label_column_options = ['Output', 'output', 'Good/Bad', 'goodbad']
        
        if self.logger:
            self.logger.info(f"Looking for label columns: {label_column_options}")
            self.logger.info(f"Available columns: {list(df.columns)}")
            
        for col in label_column_options:
            if col in df.columns:
                X = df.drop(col, axis=1)
                y = df[col]
                wafer_cols = [c for c in X.columns if c.strip().lower() == "wafer"]
                if wafer_cols:
                    X = X.drop(wafer_cols, axis=1)
                if self.logger:
                    self.logger.info(f"Separated features and label '{col}'.")
                    self.logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")
                    self.logger.info(f"Label distribution: {y.value_counts().to_dict()}")
                return X, y
        raise KeyError(f"None of the label columns {label_column_options} found in DataFrame.")

    def impute_missing_values(self, X):
        if self.logger:
            null_counts = X.isnull().sum()
            total_nulls = null_counts.sum()
            self.logger.info(f"Total missing values before imputation: {total_nulls}")
            if total_nulls > 0:
                self.logger.info(f"Columns with missing values: {null_counts[null_counts > 0].to_dict()}")
        
        imputer = KNNImputer()
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns)
        
        if self.logger:
            self.logger.info("Imputed missing values using KNNImputer.")
            self.logger.info(f"Missing values after imputation: {X.isnull().sum().sum()}")
        return X

    def preprocess(self, df, label_column_options=None):
        df = self.remove_wafer_column(df)
        X, y = self.separate_features_and_label(df, label_column_options)
        # DO NOT drop any columns
        X = self.impute_missing_values(X)
        return X, y

def preprocess_data(df, logger=None):
    if logger:
        logger.info(f"Columns in input DataFrame: {df.columns.tolist()}")
    preprocessor = Preprocessor(logger=logger)
    X, Y = preprocessor.preprocess(df)
    return X, Y

def train_models(input_file, logger):
    logger.info("="*50)
    logger.info("STARTING MODEL TRAINING PROCESS")
    logger.info("="*50)
    
    # 1. Load data
    logger.info(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # 2. Preprocess (remove wafer, etc.), but DO NOT impute yet
    logger.info("Starting preprocessing...")
    preprocessor = Preprocessor(logger=logger)
    df = preprocessor.remove_wafer_column(df)
    X, Y = preprocessor.separate_features_and_label(df)
    logger.info(f"Preprocessing completed. Features: {X.shape}, Labels: {Y.shape}")

    # 3. Split into train/test before imputation
    logger.info("Splitting data into train/test sets (70-30 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42, stratify=Y  # Changed from 0.2 to 0.3
    )
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    logger.info(f"Train labels distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test labels distribution: {y_test.value_counts().to_dict()}")

    # 4. Impute missing values (fit on train, transform both)
    logger.info("Imputing missing values...")
    imputer = KNNImputer()
    logger.info("Fitting imputer on training data...")
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    logger.info("Transforming test data...")
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
    logger.info("Imputation completed successfully")

    # 5. Clustering (fit on train, predict for all)
    logger.info("Starting clustering process...")
    kmeans = KMeansClustering()
    num_clusters = kmeans.elbow_plot(X_train_imputed, logger)
    logger.info("Assigning clusters to training data...")
    X_train_imputed['Cluster'] = kmeans.create_clusters(X_train_imputed, num_clusters, logger)
    logger.info("Assigning clusters to test data...")
    X_test_imputed['Cluster'] = kmeans.create_clusters(X_test_imputed, num_clusters, logger)

    # 6. Train and save models for each cluster
    logger.info("Starting model training for each cluster...")
    model_finder = Model_Finder(file_object=None, logger_object=logger)
    model_save_dir = os.getenv("MODEL_SAVE_DIR", "training_model")
    os.makedirs(model_save_dir, exist_ok=True)
    logger.info(f"Models will be saved to: {model_save_dir}")

    for cluster in X_train_imputed['Cluster'].unique():
        logger.info(f"Training model for cluster {cluster}...")
        train_idx = X_train_imputed['Cluster'] == cluster
        test_idx = X_test_imputed['Cluster'] == cluster

        cluster_X_train = X_train_imputed[train_idx].drop(['Cluster'], axis=1)
        cluster_y_train = y_train[train_idx].replace(-1, 0)
        cluster_X_test = X_test_imputed[test_idx].drop(['Cluster'], axis=1)
        cluster_y_test = y_test[test_idx].replace(-1, 0)

        logger.info(f"Cluster {cluster} - Train samples: {len(cluster_X_train)}, Test samples: {len(cluster_X_test)}")
        logger.info(f"Cluster {cluster} - Features: {cluster_X_train.shape[1]}")

        best_model_name, best_model = model_finder.get_best_model(
            cluster_X_train, cluster_y_train, cluster_X_test, cluster_y_test
        )

        model_filename = os.path.join(model_save_dir, f"{os.getenv('MODEL_CLUSTER_PREFIX', 'model_cluster_')}{cluster}_{best_model_name}.joblib")
        joblib.dump(best_model, model_filename)
        logger.info(f"✅ Saved {best_model_name} for cluster {cluster} as {model_filename}")

    # SAVE THE CLUSTERER after creating clusters
    os.makedirs(model_save_dir, exist_ok=True)
    clusterer_file = os.getenv('KMEANS_CLUSTERER_FILE', 'kmeans_clusterer.joblib')
    clusterer_path = os.path.join(model_save_dir, clusterer_file)
    joblib.dump(kmeans.clusterer, clusterer_path)
    logger.info(f"Saved KMeans clusterer to {clusterer_path}")

    logger.info("="*50)
    logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*50)

if __name__ == "__main__":
    # Setup logger
    logger = setup_logger()
    
    # Get input file from environment variables
    exported_dir = os.getenv('EXPORTED_DATA_DIR', 'src/exported_data_from_db')
    training_exported_file = os.getenv('TRAINING_EXPORTED_FILE', 'training_exported_data.csv')
    input_file = os.path.join(exported_dir, training_exported_file)
    
    if os.path.exists(input_file):
        train_models(input_file, logger)
    else:
        logger.error(f"Input file not found: {input_file}")
        logger.info("Please provide a valid CSV file path")

