from pathlib import Path
import logging
import pandas as pd
from src.model_creation import RandomForestModel
from src.preprocessing import Preprocessor

logger = logging.getLogger(__name__)

class DiseasePredictor:
    def __init__(self, model_path: str | Path = "model/predict_model.pkl", label_path: str | Path = "model/label.pkl") -> None:
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