from pathlib import Path
import logging
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


class ModelEvaluation:
    def __init__(self, model_path: str | Path = "model/predict_model.pkl") -> None:
        self._model = joblib.load(Path(model_path))

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        logger.info("Evaluating the trained RandomForest Model")

        prediction = self._model.predict(X_test)

        accuracy = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction, average="macro", zero_division=0)
        recall = recall_score(y_test, prediction, average="macro",zero_division=0)
        f1 = f1_score(y_test, prediction,average="macro", zero_division=0)
        report = classification_report(y_test, prediction, zero_division=0)

        logger.info(f"Accuracy : {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall   : {recall:.4f}")
        logger.info(f"F1 Score : {f1:.4f}")

        report_path = Path("model/evaluation_report.txt")
        report_path.write_text(report)

        logger.info(f"Evaluation report saved to '{report_path}'")
        logger.info("Model evaluation completed")