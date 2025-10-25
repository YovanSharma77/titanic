import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.components.preprocessing import DataPreprocessingConfig, Preprocessing

class DataLoader:
    def __init__(self,train_path:str, test_path: str):
        '''
        Initializes the Dataloader with paths to the training and testing datasets
        '''
        self.train_path = train_path
        self.test_path = test_path
        
    def load_data(self):
        '''
        Loads train and test datasets as pandas Dataframes.
        Raises CustomException if any issue occurs.
        '''
        try:
            logging.info('Started loading datasets...')
            
            #validate file paths
            if not os.path.exists(self.train_path):
                raise FileNotFoundError(f'Train file not found at {self.train_path}')
            if not os.path.exists(self.test_path):
                raise FileNotFoundError(f'test file not found at {self.test_path}')
            
            #load data
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
            
            logging.info(f'Train data loaded successfully with shape: {train_df.shape}')
            logging.info(f'test data loaded successfully with shape: {test_df.shape}')
            
            return train_df, test_df
        
        except Exception as e:
            logging.error('Error occured while loading data')
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    train_file  = os.path.join('data','train_cleaned.csv')
    test_file = os.path.join('data','test_cleaned.csv')
    
    loader = DataLoader(train_file,test_file)
    train_df, test_df = loader.load_data()
    
    preprocessing = Preprocessing()
    
    train_arr, test_arr, preprocessor_path = preprocessing.initiate_data_preprocessing(
        train_path=train_file,
        test_path=test_file
    )
    
    print("Preprocessing completed successfully!")
    print(f"Preprocessor object saved at: {preprocessor_path}")