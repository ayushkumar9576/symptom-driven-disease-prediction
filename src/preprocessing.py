from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, minimum_samples: int = 10) -> None:
        self._minimum_samples = minimum_samples
        self._label_encoder = LabelEncoder()

    @staticmethod
    def _remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates().reset_index(drop=True)

    @staticmethod
    def _remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
        constant_columns = [column for column in df.columns[1:] if df[column].nunique() == 1]
        return df.drop(columns=constant_columns)

    def _remove_rare_diseases(self, df: pd.DataFrame) -> pd.DataFrame:
        disease_counts = df["diseases"].value_counts()
        valid_diseases = disease_counts[disease_counts >= self._minimum_samples].index
        return (df[df["diseases"].isin(valid_diseases)].reset_index(drop=True))

    def _encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["diseases"] = self._label_encoder.fit_transform(df["diseases"])
        return df

    def save_label_encoder(self, path: str | Path) -> None:
        joblib.dump(self._label_encoder, Path(path))
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing on the Dataset")
        df = self._remove_duplicate_rows(df)
        df = self._remove_rare_diseases(df)
        df = self._remove_constant_columns(df)
        df = self._encode_target(df)
        self.save_label_encoder("model/lable.pkl")
        logger.info("Preprocessing on the data completed")
        return df