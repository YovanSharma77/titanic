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
    preprocessor_obj_file_path = os.path.join('Data','preprocessor.pkl')

class Preprocessing:
    def __init__(self):
        self.data_preprocessing_config = DataPreprocessingConfig()
    def get_data_preprocesser_object(self):
        try:
            num_feature = ['Age','Fare']
            binary_feature = ["Sex",'Sibsp','Parch']
            multi_feature = ["Embarked", "Pclass"]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                    ('scaler', StandardScaler())
                ]
            )
            binary_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent"))
                    ("label_encoder", OrdinalEncoder())  
                ])   
            multi_cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent"))
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
                ])
            
            logging.info(f"Categorical columns: {binary_feature, multi_feature}")
            logging.info(f"Numerical fea: {num_feature}")
            
            preprocessor = ColumnTransformer(
                [("num_pipeline", num_pipeline, num_feature),
                ("binary_pipeline", binary_pipeline, binary_feature),
                ("multi_cat_pipeline", multi_cat_pipeline, multi_feature)
                ])
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_preprocessing(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Reading train and test data completed')
            logging.info('obtaining preprocessing object')
            
            preprocessing_obj = self.get_data_preprocesser_object()
            
            target_column = 'Survived'
            
            input_feature_train_df = train_df.drop(columns=[target_column],axis=1)
            input_feature_test_df = test_df.drop(columns=[target_column],axis=1)
            
            target_feature_train_df = train_df[target_column]
            target_feature_test_df = test_df[target_column]
            
            logging.info('Applying preprocessing object on training and testing dataframe')
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info('Saved preprocessed file')
            
            save_object(
                file_path=self.data_preprocessing_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_preprocessing_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)