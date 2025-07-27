import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

class Model_Finder:
    """
    This class is used to find the best machine learning model for the given data.
    """
    
    def __init__(self, file_object=None, logger_object=None):
        self.file_object = file_object
        self.logger_object = logger_object
        
    def get_best_params_for_random_forest(self, train_x, train_y):
        """
        Method to get the best parameters for Random Forest Algorithm
        """
        try:
            if self.logger_object:
                self.logger_object.info('Entered the get_best_params_for_random_forest method')
            
            # Simple parameter grid for quick results
            param_grid = {
                'n_estimators': [100],
                'max_depth': [10, None],
                'class_weight': ['balanced']  # Add class balancing!
            }
            
            grid = GridSearchCV(RandomForestClassifier(random_state=42), 
                            param_grid, cv=3, scoring='f1')
            grid.fit(train_x, train_y)
            
            if self.logger_object:
                self.logger_object.info('Best parameters for Random Forest: ' + str(grid.best_params_))
                self.logger_object.info('Exited the get_best_params_for_random_forest method')
            
            return grid.best_params_
            
        except Exception as e:
            if self.logger_object:
                self.logger_object.error('Exception occurred in get_best_params_for_random_forest method: ' + str(e))
            # Return default parameters as fallback
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'class_weight': 'balanced',
                'random_state': 42
            }

    def get_best_params_for_gradient_boosting(self, train_x, train_y):
        """
        Method to get the best parameters for Gradient Boosting Algorithm
        """
        try:
            if self.logger_object:
                self.logger_object.info('Entered the get_best_params_for_gradient_boosting method')
            
            # Using default parameters for simplicity
            param_grid = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            }
            
            if self.logger_object:
                self.logger_object.info('Best parameters for Gradient Boosting: ' + str(param_grid))
                self.logger_object.info('Exited the get_best_params_for_gradient_boosting method')
            
            return param_grid
            
        except Exception as e:
            if self.logger_object:
                self.logger_object.error('Exception occurred in get_best_params_for_gradient_boosting method: ' + str(e))
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            }

    def get_best_params_for_svm(self, train_x, train_y):
        """
        Method to get the best parameters for SVM Algorithm
        """
        try:
            if self.logger_object:
                self.logger_object.info('Entered the get_best_params_for_svm method')
            
            # Using default parameters for simplicity
            param_grid = {
                'C': 1.0,
                'kernel': 'rbf',
                'class_weight': 'balanced',
                'random_state': 42
            }
            
            if self.logger_object:
                self.logger_object.info('Best parameters for SVM: ' + str(param_grid))
                self.logger_object.info('Exited the get_best_params_for_svm method')
            
            return param_grid
            
        except Exception as e:
            if self.logger_object:
                self.logger_object.error('Exception occurred in get_best_params_for_svm method: ' + str(e))
            return {
                'C': 1.0,
                'kernel': 'rbf',
                'class_weight': 'balanced',
                'random_state': 42
            }

    def get_best_model(self, train_x, train_y, test_x, test_y):
        """
        Method to find the best machine learning model
        """
        try:
            if self.logger_object:
                self.logger_object.info('Entered the get_best_model method')
            
            # Dictionary to store model results
            model_results = {}
            
            # Random Forest
            if self.logger_object:
                self.logger_object.info('Training Random Forest')
            rf_params = self.get_best_params_for_random_forest(train_x, train_y)
            rf_model = RandomForestClassifier(**rf_params)
            rf_model.fit(train_x, train_y)
            rf_pred = rf_model.predict(test_x)
            rf_accuracy = accuracy_score(test_y, rf_pred)
            rf_f1 = f1_score(test_y, rf_pred, average='weighted')
            model_results['RandomForest'] = {'model': rf_model, 'accuracy': rf_accuracy, 'f1': rf_f1}
            
            if self.logger_object:
                self.logger_object.info(f'Random Forest - Accuracy: {rf_accuracy}, F1: {rf_f1}')
            
            # Gradient Boosting
            if self.logger_object:
                self.logger_object.info('Training Gradient Boosting')
            gb_params = self.get_best_params_for_gradient_boosting(train_x, train_y)
            gb_model = GradientBoostingClassifier(**gb_params)
            gb_model.fit(train_x, train_y)
            gb_pred = gb_model.predict(test_x)
            gb_accuracy = accuracy_score(test_y, gb_pred)
            gb_f1 = f1_score(test_y, gb_pred, average='weighted')
            model_results['GradientBoosting'] = {'model': gb_model, 'accuracy': gb_accuracy, 'f1': gb_f1}
            
            if self.logger_object:
                self.logger_object.info(f'Gradient Boosting - Accuracy: {gb_accuracy}, F1: {gb_f1}')
            
            # Logistic Regression
            if self.logger_object:
                self.logger_object.info('Training Logistic Regression')
            lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
            lr_model.fit(train_x, train_y)
            lr_pred = lr_model.predict(test_x)
            lr_accuracy = accuracy_score(test_y, lr_pred)
            lr_f1 = f1_score(test_y, lr_pred, average='weighted')
            model_results['LogisticRegression'] = {'model': lr_model, 'accuracy': lr_accuracy, 'f1': lr_f1}
            
            if self.logger_object:
                self.logger_object.info(f'Logistic Regression - Accuracy: {lr_accuracy}, F1: {lr_f1}')
            
            # Decision Tree
            if self.logger_object:
                self.logger_object.info('Training Decision Tree')
            dt_model = DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced')
            dt_model.fit(train_x, train_y)
            dt_pred = dt_model.predict(test_x)
            dt_accuracy = accuracy_score(test_y, dt_pred)
            dt_f1 = f1_score(test_y, dt_pred, average='weighted')
            model_results['DecisionTree'] = {'model': dt_model, 'accuracy': dt_accuracy, 'f1': dt_f1}
            
            if self.logger_object:
                self.logger_object.info(f'Decision Tree - Accuracy: {dt_accuracy}, F1: {dt_f1}')
            
            # Try XGBoost if available
            try:
                import xgboost as xgb
                if self.logger_object:
                    self.logger_object.info('Training XGBoost')
                
                # Calculate positive class weight for imbalanced data
                pos_weight = len(train_y[train_y == 0]) / max(len(train_y[train_y == 1]), 1)
                
                xgb_model = xgb.XGBClassifier(
                    learning_rate=0.05,
                    n_estimators=100,
                    max_depth=5,
                    scale_pos_weight=pos_weight,
                    random_state=42
                )
                xgb_model.fit(train_x, train_y)
                xgb_pred = xgb_model.predict(test_x)
                xgb_accuracy = accuracy_score(test_y, xgb_pred)
                xgb_f1 = f1_score(test_y, xgb_pred, average='weighted')
                model_results['XGBoost'] = {'model': xgb_model, 'accuracy': xgb_accuracy, 'f1': xgb_f1}
                
                if self.logger_object:
                    self.logger_object.info(f'XGBoost - Accuracy: {xgb_accuracy}, F1: {xgb_f1}')
                    
            except Exception as e:
                if self.logger_object:
                    self.logger_object.info(f'XGBoost not available or failed: {str(e)}')
            
            # Find the best model using F1 score (better for imbalanced data)
            best_model_name = max(model_results, key=lambda x: model_results[x]['f1'])
            best_model = model_results[best_model_name]['model']
            best_accuracy = model_results[best_model_name]['accuracy']
            best_f1 = model_results[best_model_name]['f1']
            
            if self.logger_object:
                self.logger_object.info(f'Best model: {best_model_name} with accuracy: {best_accuracy}, F1: {best_f1}')
                self.logger_object.info('Exited the get_best_model method')
            
            return best_model_name, best_model
            
        except Exception as e:
            if self.logger_object:
                self.logger_object.error('Exception occurred in get_best_model method: ' + str(e))
            raise Exception()