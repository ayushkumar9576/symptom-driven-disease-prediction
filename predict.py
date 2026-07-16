from pathlib import Path
import logging
import pandas as pd
from src.model_creation import RandomForestModel
from src.preprocessing import Preprocessor

logger = logging.getLogger(__name__)

class DiseasePredictor:
    def __init__(self, model_path: str | Path = "model/predict_model.pkl", label_path: str | Path = "model/lable.pkl") -> None:
        logger.info("Loading trained model and label encoder")
        self._model = RandomForestModel.load(model_path)
        self._label_encoder = Preprocessor.load_label_encoder(label_path)

    def predict(self, patient: pd.DataFrame) -> str:
        logger.info("Predicting disease Based on Symptoms")
        prediction = self._model.predict(patient)
        return self._label_encoder.inverse_transform([prediction])[0]

    def predict_probability(self, patient: pd.DataFrame) -> float:
        probability = self._model.predict_proba(patient)
        return float(probability.max())
    
    def predict_top_predictions(self, patient: pd.DataFrame, top_k: int = 3) -> list[tuple[str, float]]:
        probability = self._model.predict_proba(patient)[0]
        classes = self._label_encoder.classes_
        predictions = sorted(zip(classes, probability), key=lambda item: item[1], reverse=True)

        return [(disease, float(score)) for disease, score in predictions[:top_k]]
    
    def feature_names(self) -> list[str]:
        return self._model.feature_names()