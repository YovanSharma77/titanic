import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataPreprocessingConfig:
    preprocessor_obj_file_path = os.path.join('Data', 'preprocessing.pkl')

class Preprocessing:
    def __init__(self):
        self.data_preprocessing_config = DataPreprocessingConfig()

    def get_data_preprocessor_object(self):
        try:
            num_features = ['Age', 'Fare', 'SibSp', 'Parch']
            binary_features = ["Sex"]
            multi_features = ["Embarked", "Pclass"]
            
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ('scaler', StandardScaler())
            ])
            
            binary_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder())
            ])
            
            multi_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            
            logging.info(f"Numerical features: {num_features}")
            logging.info(f"Binary features: {binary_features}")
            logging.info(f"Multi-class features: {multi_features}")
            
            
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_features),
                ("binary_pipeline", binary_pipeline, binary_features),
                ("multi_pipeline", multi_pipeline, multi_features)
            ])
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_preprocessing(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Reading train and test data completed.')
            
            preprocessing_obj = self.get_data_preprocessor_object()
            
            target_column = 'Survived'
            
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.copy()
            target_feature_test_df = test_df[target_column] if target_column in test_df.columns else None

            logging.info('Applying preprocessing object on training and testing dataframes.')
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            
            if target_feature_test_df is not None:
                 test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            else: 
                 test_arr = input_feature_test_arr

            logging.info('Saved preprocessing object.')
            
            save_object(
                file_path=self.data_preprocessing_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_preprocessing_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)