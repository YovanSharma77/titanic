import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.preprocessing import Preprocessing
from src.components.model import ModelTrainer

if __name__ == "__main__":
    try:
        logging.info("Pipeline execution started...")

        # --- STEP 1: Define File Paths ---
        train_file = os.path.join('data', 'train.csv')
        test_file  = os.path.join('data', 'test.csv')
        
        # --- STEP 2: Data Preprocessing ---
        logging.info("Starting data preprocessing...")
        preprocessing = Preprocessing()
        train_arr, test_arr, preprocessor_path = preprocessing.initiate_data_preprocessing(
            train_path=train_file,
            test_path=test_file
        )
        print("✅ Preprocessing completed successfully!")
        print(f"Preprocessor object saved at: {preprocessor_path}")
        logging.info("Data preprocessing completed successfully!")

        # --- STEP 3: Model Training ---
        logging.info("Starting model training...")
        model_trainer = ModelTrainer()
        
        from sklearn.model_selection import train_test_split
        train_set, test_set = train_test_split(train_arr, test_size=0.2, random_state=42)

        f1score = model_trainer.initiate_model_trainer(train_set, test_set)

        print(f"✅ Model training completed successfully!")
        print(f"Best Model F1 Score on Test Data: {f1score:.4f}")
        logging.info(f"Model training completed successfully with F1 Score: {f1score:.4f}")

        logging.info("Pipeline execution finished successfully!")

    except Exception as e:
        logging.exception("Error occurred during pipeline execution")
        raise CustomException(e, sys)