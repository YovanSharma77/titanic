import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

class ModelTrainerConfig:
    trained_model_file_path = os.path.join('Data', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting train and test input data.')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear'),
                'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
                'Naive Bayes (Gaussian)': GaussianNB(),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                'LightGBM': LGBMClassifier(verbose=-1),
                'CatBoost': CatBoostClassifier(verbose=0)
            }
            
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train,
                                                 X_test=X_test, y_test=y_test, models=models)
            
            best_model_name = max(model_report, key=lambda name: model_report[name]['Test F1 Score'])
            best_model_score = model_report[best_model_name]['Test F1 Score']
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No model found with F1 score above 0.6", sys)
            
            logging.info(f"Best Model Found: {best_model_name} | F1 Score: {best_model_score:.4f}")

            save_object(
                file_path=self.model_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            final_f1_score = f1_score(y_test, predicted, average='weighted')
            
            return final_f1_score
            
        except Exception as e:
            raise CustomException(e, sys)