import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from src.data_preprocessing.clustering import KMeansClustering
from src.best_model_finder.tuner import Model_Finder
import joblib
import os

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
        for col in label_column_options:
            if col in df.columns:
                X = df.drop(col, axis=1)
                y = df[col]
                # Remove wafer column(s) from X if present
                wafer_cols = [c for c in X.columns if c.strip().lower() == "wafer"]
                if wafer_cols:
                    X = X.drop(wafer_cols, axis=1)
                    if self.logger:
                        self.logger.info(f"Removed wafer column(s) from features: {wafer_cols}")
                if self.logger:
                    self.logger.info(f"Separated features and label '{col}'.")
                return X, y
        if self.logger:
            self.logger.error(f"None of the label columns {label_column_options} found in DataFrame. Columns are: {df.columns.tolist()}")
        raise KeyError(f"None of the label columns {label_column_options} found in DataFrame. Columns are: {df.columns.tolist()}")

    def remove_zero_std_columns(self, X):
        zero_std_cols = X.columns[X.std() == 0]
        X = X.drop(zero_std_cols, axis=1)
        if self.logger:
            self.logger.info(f"Removed zero std columns: {list(zero_std_cols)}")
        return X

    def impute_missing_values(self, X):
        imputer = KNNImputer()
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns)
        if self.logger:
            self.logger.info("Imputed missing values using KNNImputer.")
        return X

    def preprocess(self, df, label_column_options=None):
        df = self.remove_wafer_column(df)
        X, y = self.separate_features_and_label(df, label_column_options)
        X = self.remove_zero_std_columns(X)
        X = self.impute_missing_values(X)
        return X, y

def preprocess_data(df, logger=None):
    if logger:
        logger.info(f"Columns in input DataFrame: {df.columns.tolist()}")
    preprocessor = Preprocessor(logger=logger)
    X, Y = preprocessor.preprocess(df)
    return X, Y

def train_models(input_file, logger):
    # 1. Load data
    df = pd.read_csv(input_file)
    # 2. Preprocess (remove wafer, impute, etc.)
    X, Y = preprocess_data(df, logger=logger)
    # 3. Clustering
    kmeans = KMeansClustering()
    num_clusters = kmeans.elbow_plot(X, logger)
    X['Cluster'] = kmeans.create_clusters(X, num_clusters, logger)
    # 4. Train and save models for each cluster
    model_finder = Model_Finder(file_object=None, logger_object=logger)

    # Get model save directory from environment
    model_save_dir = os.getenv("MODEL_SAVE_DIR", "training_model")
    os.makedirs(model_save_dir, exist_ok=True)

    for cluster in X['Cluster'].unique():
        cluster_data = X[X['Cluster'] == cluster]
        cluster_features = cluster_data.drop(['Cluster'], axis=1)
        cluster_labels = Y[cluster_data.index].replace(-1, 0)  # Ensure labels are 0/1

        best_model_name, best_model = model_finder.get_best_model(
            cluster_features, cluster_labels, cluster_features, cluster_labels
        )

        # Save the model for this cluster in the specified directory
        model_filename = os.path.join(model_save_dir, f"model_cluster_{cluster}_{best_model_name}.joblib")
        joblib.dump(best_model, model_filename)
        logger.info(f"Saved {best_model_name} for cluster {cluster} as {model_filename}")

    logger.info("Training and saving models for all clusters completed.")

