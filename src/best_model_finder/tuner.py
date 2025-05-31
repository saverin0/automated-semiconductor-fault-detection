from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

class Model_Finder:
    """
    This class shall be used to find the model with best accuracy and AUC score.
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def get_best_model(self, train_x, train_y, test_x, test_y):
        self.logger_object.info('Entered the get_best_model method of the Model_Finder class')
        try:
            # Convert -1 labels to 0 for binary classification
            train_y = train_y.replace(-1, 0)
            test_y = test_y.replace(-1, 0)

            # Feature scaling
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)

            models = {
                "RandomForest": RandomForestClassifier(),
                "XGBoost": XGBClassifier(eval_metric='logloss'),
                "LogisticRegression": LogisticRegression(max_iter=2000),
                "DecisionTree": DecisionTreeClassifier(),
                "SVM": SVC(probability=True),
                "KNN": KNeighborsClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier()
            }
            best_score = 0
            best_model = None
            best_model_name = None

            for name, model in models.items():
                model.fit(train_x, train_y)
                preds = model.predict(test_x)
                # Use AUC if possible, else accuracy
                if len(test_y.unique()) == 1:
                    score = accuracy_score(test_y, preds)
                    self.logger_object.info(f"Accuracy for {name}: {score:.4f}")
                else:
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(test_x)[:, 1]
                    else:
                        proba = preds  # fallback
                    score = roc_auc_score(test_y, proba)
                    self.logger_object.info(f"AUC for {name}: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name

            self.logger_object.info(f"Best model: {best_model_name} with score: {best_score:.4f}")
            return best_model_name, best_model

        except Exception as e:
            self.logger_object.error('Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.error('Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()

