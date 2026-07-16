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

    def save_label_encoder(self, path: str | Path = "model/lable.pkl") -> None:
        joblib.dump(self._label_encoder, Path(path))
    
    @staticmethod
    def load_label_encoder(path: str | Path = "model/lable.pkl") -> LabelEncoder:
        return joblib.load(Path(path))

    def _reduce_dataset_size(self, df: pd.DataFrame, target_rows: int = 100000) -> pd.DataFrame:
        if len(df) <= target_rows:
            return df.reset_index(drop=True)

        fraction = target_rows / len(df)
        reduced_groups = []
        for _, group in df.groupby("diseases"):
            sample_size = max(1, round(len(group) * fraction))
            reduced_groups.append(group.sample(n=sample_size, random_state=42))

        return pd.concat(reduced_groups, ignore_index=True)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing on the Dataset")
        df = self._remove_duplicate_rows(df)
        df = self._remove_rare_diseases(df)
        df = self._reduce_dataset_size(df, target_rows= 150000)
        df = self._remove_constant_columns(df)
        df = self._encode_target(df)
        self.save_label_encoder("model/lable.pkl")
        logger.info("Preprocessing on the data completed")
        return df