import os
import sys
import pickle
from src.exception import CustomException
from sklearn.metrics import f1_score 

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            f1_train = f1_score(y_train, y_train_pred, average='weighted')
            f1_test = f1_score(y_test, y_test_pred, average='weighted')

            report[name] = {
                'Train F1 Score': f1_train,
                'Test F1 Score': f1_test
            }
        return report

    except Exception as e:
        raise CustomException(e, sys)