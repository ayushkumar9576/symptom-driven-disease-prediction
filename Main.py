import logging
from src.data_loader import DataLoader
from src.data_splitter import DataSplitter
from src.model_creation import RandomForestModel
from src.model_evaluator import ModelEvaluation
from src.preprocessing import Preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TRAIN_MODEL = False

def main()-> None:
    loader = DataLoader("data/archive.zip")
    loader.save()
    df = loader.load("data/Final_Augmented_dataset_Diseases_and_Symptoms.csv")

    preprocessor = Preprocessor(10)
    df = preprocessor.preprocess(df)

    splitter = DataSplitter(train_size= 0.7, validation_size= 0.1, test_size= 0.2, random_state= 42)
    X_train, X_validate, X_test, y_train, y_validate, y_test = splitter.split(df= df)

    model = RandomForestModel("model/predict_model.pkl")
    if TRAIN_MODEL:
        model.train(X_train, y_train, X_validate, y_validate)
    
    evaluator = ModelEvaluation("model/predict_model.pkl")
    evaluator.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()