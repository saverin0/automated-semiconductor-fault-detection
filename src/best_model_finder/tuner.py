import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class Model_Finder:
    """
    Optimized class to find the best machine learning model with proper hyperparameter tuning.
    """
    
    def __init__(self, file_object, logger_object, use_quick_mode=True, ultra_fast_mode=True):
        self.file_object = file_object
        self.logger_object = logger_object
        self.clf = RandomForestClassifier()
        self.xgb = XGBClassifier()
        self.svc = SVC()
        self.nb = GaussianNB()
        self.knn = KNeighborsClassifier()
        self.lgb = None
        self.use_quick_mode = use_quick_mode
        self.ultra_fast_mode = ultra_fast_mode  # Even more aggressive optimization
        
        # Try to import LightGBM if available
        try:
            import lightgbm as lgb
            self.lgb = lgb.LGBMClassifier()
        except ImportError:
            self.logger_object.info("LightGBM not available: No module named 'lightgbm'")
            pass
        
    def get_best_params_for_random_forest(self, train_x, train_y):
        """
        Optimized method to get the best parameters for Random Forest Algorithm
        """
        try:
            if self.logger_object:
                self.logger_object.info('Entered the get_best_params_for_random_forest method')
            
            if self.ultra_fast_mode:
                # Ultra-fast mode for small datasets - minimal parameters
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [10],
                    'min_samples_split': [2],
                    'min_samples_leaf': [1],
                    'class_weight': ['balanced']
                }
                search_method = GridSearchCV
                cv_folds = 2  # Minimal CV
            elif self.use_quick_mode:
                # Quick mode with limited search
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'class_weight': ['balanced']
                }
                search_method = GridSearchCV
                cv_folds = 3
            else:
                # Comprehensive parameter grid for thorough search
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt', 'log2'],
                    'bootstrap': [True, False],
                    'class_weight': ['balanced', 'balanced_subsample']
                }
                # Use RandomizedSearchCV for faster search with more parameters
                search_method = RandomizedSearchCV
                cv_folds = 5
            
            if search_method == RandomizedSearchCV:
                grid = search_method(
                    RandomForestClassifier(random_state=42, n_jobs=-1),
                    param_grid, 
                    n_iter=10,  # Reduced iterations for faster training
                    cv=cv_folds,  # Use variable cv_folds
                    scoring='f1_weighted',
                    n_jobs=-1,
                    random_state=42
                )
            else:
                grid = search_method(
                    RandomForestClassifier(random_state=42, n_jobs=-1),
                    param_grid, 
                    cv=cv_folds,  # Use variable cv_folds
                    scoring='f1_weighted',
                    n_jobs=-1
                )
            
            grid.fit(train_x, train_y)
            
            best_params = grid.best_params_
            best_params['random_state'] = 42
            best_params['n_jobs'] = -1
            
            if self.logger_object:
                self.logger_object.info('Best parameters for Random Forest: ' + str(best_params))
                self.logger_object.info('Best CV score: ' + str(grid.best_score_))
                self.logger_object.info('Exited the get_best_params_for_random_forest method')
            
            return best_params
            
        except Exception as e:
            if self.logger_object:
                self.logger_object.error('Exception occurred in get_best_params_for_random_forest method: ' + str(e))
            # Return optimized default parameters
            return {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }

    def get_best_params_for_gradient_boosting(self, train_x, train_y):
        """
        Optimized method to get the best parameters for Gradient Boosting Algorithm
        """
        try:
            if self.logger_object:
                self.logger_object.info('Entered the get_best_params_for_gradient_boosting method')
            
            if self.ultra_fast_mode:
                param_grid = {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1],
                    'max_depth': [3],
                    'subsample': [1.0]
                }
                cv_folds = 2
                search_method = GridSearchCV
            elif self.use_quick_mode:
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                }
                cv_folds = 3
                search_method = GridSearchCV
            else:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'subsample': [0.7, 0.8, 0.9],
                    'max_features': ['sqrt', 'log2']
                }
                cv_folds = 5
                search_method = RandomizedSearchCV
            
            # Use appropriate search method based on mode
            if search_method == RandomizedSearchCV:
                grid = RandomizedSearchCV(
                    GradientBoostingClassifier(random_state=42),
                    param_grid,
                    n_iter=6 if self.ultra_fast_mode else 10,  # Even fewer iterations for ultra-fast
                    cv=cv_folds,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    random_state=42
                )
            else:
                grid = GridSearchCV(
                    GradientBoostingClassifier(random_state=42),
                    param_grid,
                    cv=cv_folds,
                    scoring='f1_weighted',
                    n_jobs=-1
                )
            
            grid.fit(train_x, train_y)
            
            best_params = grid.best_params_
            best_params['random_state'] = 42
            
            if self.logger_object:
                self.logger_object.info('Best parameters for Gradient Boosting: ' + str(best_params))
                self.logger_object.info('Best CV score: ' + str(grid.best_score_))
                self.logger_object.info('Exited the get_best_params_for_gradient_boosting method')
            
            return best_params
            
        except Exception as e:
            if self.logger_object:
                self.logger_object.error('Exception occurred in get_best_params_for_gradient_boosting method: ' + str(e))
            return {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'subsample': 0.8,
                'max_features': 'sqrt',
                'random_state': 42
            }

    def get_best_params_for_xgboost(self, train_x, train_y):
        """
        Comprehensive parameter tuning for XGBoost - the most important model
        """
        try:
            import xgboost as xgb
            
            if self.logger_object:
                self.logger_object.info('Entered the get_best_params_for_xgboost method')
            
            # Calculate scale_pos_weight for imbalanced data
            pos_weight = len(train_y[train_y == 0]) / max(len(train_y[train_y == 1]), 1)
            
            if self.ultra_fast_mode:
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [3],
                    'learning_rate': [0.1],
                    'subsample': [1.0],
                    'colsample_bytree': [1.0],
                    'scale_pos_weight': [pos_weight]
                }
                cv_folds = 2
                search_method = GridSearchCV
            elif self.use_quick_mode:
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                    'scale_pos_weight': [1, pos_weight]
                }
                cv_folds = 3
                search_method = GridSearchCV
            else:
                # Comprehensive parameter grid for XGBoost
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9],
                    'min_child_weight': [1, 3, 5],
                    'gamma': [0, 0.1, 0.2],
                    'reg_alpha': [0, 0.01, 0.1],
                    'reg_lambda': [0, 0.01, 0.1],
                    'scale_pos_weight': [1, pos_weight]
                }
                cv_folds = 5
                search_method = RandomizedSearchCV
            
            # Use appropriate search method based on mode
            if search_method == RandomizedSearchCV:
                grid = RandomizedSearchCV(
                    xgb.XGBClassifier(
                        random_state=42,
                        n_jobs=-1,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    ),
                    param_grid,
                    n_iter=4 if self.ultra_fast_mode else 10,  # Even fewer iterations for ultra-fast
                    cv=cv_folds,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    random_state=42,
                    verbose=0
                )
            else:
                grid = GridSearchCV(
                    xgb.XGBClassifier(
                        random_state=42,
                        n_jobs=-1,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    ),
                    param_grid,
                    cv=cv_folds,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=0
                )
            
            grid.fit(train_x, train_y)
            
            best_params = grid.best_params_
            best_params['random_state'] = 42
            best_params['n_jobs'] = -1
            best_params['use_label_encoder'] = False
            best_params['eval_metric'] = 'logloss'
            
            if self.logger_object:
                self.logger_object.info('Best parameters for XGBoost: ' + str(best_params))
                self.logger_object.info('Best CV score: ' + str(grid.best_score_))
                self.logger_object.info('Exited the get_best_params_for_xgboost method')
            
            return best_params
            
        except Exception as e:
            if self.logger_object:
                self.logger_object.error('Exception occurred in get_best_params_for_xgboost method: ' + str(e))
            
            # Return optimized default parameters
            pos_weight = len(train_y[train_y == 0]) / max(len(train_y[train_y == 1]), 1)
            return {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.01,
                'reg_lambda': 0.01,
                'scale_pos_weight': pos_weight,
                'random_state': 42,
                'n_jobs': -1,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }

    def get_best_params_for_svm(self, train_x, train_y):
        """
        Optimized method to get the best parameters for SVM Algorithm
        """
        try:
            if self.logger_object:
                self.logger_object.info('Entered the get_best_params_for_svm method')
            
            # Scale features for SVM (important!)
            scaler = StandardScaler()
            train_x_scaled = scaler.fit_transform(train_x)
            
            if self.use_quick_mode:
                param_grid = {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale'],
                    'class_weight': ['balanced']
                }
            else:
                param_grid = {
                    'C': [0.01, 0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto'],
                    'class_weight': ['balanced', None]
                }
            
            grid = GridSearchCV(
                SVC(random_state=42, probability=True),
                param_grid,
                cv=3,
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            grid.fit(train_x_scaled, train_y)
            
            best_params = grid.best_params_
            best_params['random_state'] = 42
            best_params['probability'] = True
            
            # Store scaler for later use
            self.svm_scaler = scaler
            
            if self.logger_object:
                self.logger_object.info('Best parameters for SVM: ' + str(best_params))
                self.logger_object.info('Best CV score: ' + str(grid.best_score_))
                self.logger_object.info('Exited the get_best_params_for_svm method')
            
            return best_params
            
        except Exception as e:
            if self.logger_object:
                self.logger_object.error('Exception occurred in get_best_params_for_svm method: ' + str(e))
            return {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'class_weight': 'balanced',
                'random_state': 42,
                'probability': True
            }

    def get_best_model(self, train_x, train_y, test_x, test_y):
        """
        Find the model with the best f1 score on validation data.
        Ultra-fast mode significantly reduces hyperparameter search for small datasets.
        """
        try:
            self.logger_object.info('Entered the get_best_model method')
            self.logger_object.info(f'Training data shape: {train_x.shape}')
            self.logger_object.info(f'Class distribution: {np.bincount(train_y)}')
            
            # For very small datasets, use simple default parameters
            if self.ultra_fast_mode and len(train_x) < 500:
                self.logger_object.info('Ultra-fast mode: Using optimized default parameters for small dataset')
                
                # Simple Random Forest with good defaults
                rf_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
                rf_model.fit(train_x, train_y)
                rf_pred = rf_model.predict(test_x)
                rf_accuracy = accuracy_score(test_y, rf_pred)
                rf_f1 = f1_score(test_y, rf_pred, average='weighted')
                try:
                    rf_auc = roc_auc_score(test_y, rf_model.predict_proba(test_x)[:, 1])
                except:
                    rf_auc = 0.5
                
                self.logger_object.info(f'Random Forest (default) - Accuracy: {rf_accuracy:.4f}, F1: {rf_f1:.4f}, AUC: {rf_auc:.4f}')
                
                return 'Random Forest', rf_model
            
            # Normal training with hyperparameter tuning
            # Training Random Forest
            self.logger_object.info('Training Random Forest with hyperparameter tuning')
            rf_params = self.get_best_params_for_random_forest(train_x, train_y)
            clf = RandomForestClassifier(**rf_params)
            clf.fit(train_x, train_y)
            
            pred_rf = clf.predict(test_x)
            
            rf_accuracy = accuracy_score(test_y, pred_rf)
            rf_f1 = f1_score(test_y, pred_rf, average='weighted')
            try:
                rf_auc = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
            except:
                rf_auc = 0.5
            
            self.logger_object.info(f'Random Forest - Accuracy: {rf_accuracy:.4f}, F1: {rf_f1:.4f}, AUC: {rf_auc:.4f}')
            
            # Training XGBoost
            self.logger_object.info('Training XGBoost with hyperparameter tuning')
            xgb_params = self.get_best_params_for_xgboost(train_x, train_y)
            xgb_clf = XGBClassifier(**xgb_params)
            xgb_clf.fit(train_x, train_y)
            
            pred_xgb = xgb_clf.predict(test_x)
            
            xgb_accuracy = accuracy_score(test_y, pred_xgb)
            xgb_f1 = f1_score(test_y, pred_xgb, average='weighted')
            try:
                xgb_auc = roc_auc_score(test_y, xgb_clf.predict_proba(test_x)[:, 1])
            except:
                xgb_auc = 0.5
            
            self.logger_object.info(f'XGBoost - Accuracy: {xgb_accuracy:.4f}, F1: {xgb_f1:.4f}, AUC: {xgb_auc:.4f}')
            
            # Training Gradient Boosting
            self.logger_object.info('Training Gradient Boosting with hyperparameter tuning')
            gb_params = self.get_best_params_for_gradient_boosting(train_x, train_y)
            gb_clf = GradientBoostingClassifier(**gb_params)
            gb_clf.fit(train_x, train_y)
            
            pred_gb = gb_clf.predict(test_x)
            
            gb_accuracy = accuracy_score(test_y, pred_gb)
            gb_f1 = f1_score(test_y, pred_gb, average='weighted')
            try:
                gb_auc = roc_auc_score(test_y, gb_clf.predict_proba(test_x)[:, 1])
            except:
                gb_auc = 0.5
            
            self.logger_object.info(f'Gradient Boosting - Accuracy: {gb_accuracy:.4f}, F1: {gb_f1:.4f}, AUC: {gb_auc:.4f}')
            
            # Compare models and return the best one
            scores = {
                'Random Forest': rf_f1,
                'XGBoost': xgb_f1,
                'Gradient Boosting': gb_f1
            }
            
            models = {
                'Random Forest': clf,
                'XGBoost': xgb_clf,
                'Gradient Boosting': gb_clf
            }
            
            best_model_name = max(scores, key=scores.get)
            best_model = models[best_model_name]
            
            self.logger_object.info(f'Best model: {best_model_name} with F1 score: {scores[best_model_name]:.4f}')
            
            return best_model_name, best_model
            
        except Exception as e:
            self.logger_object.exception(f'Exception occurred in get_best_model method: {str(e)}')
            # Return a simple default model
            default_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            default_model.fit(train_x, train_y)
            return 'Random Forest (Default)', default_model