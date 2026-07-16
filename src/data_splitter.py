from sklearn.model_selection import train_test_split
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataSplitter:
    def __init__(self, train_size: float = 0.7, validation_size: float = 0.1, test_size: float = 0.2, random_state: int = 42)-> None:
        if abs(train_size + validation_size + test_size - 1.0) > 1e-9:
            raise ValueError("train_size + validation_size + test_size must equal 1.")
        self._train_size = train_size
        self._validation_size = validation_size
        self._test_size = test_size
        self._random_state = random_state
    
    def split(self, df: pd.DataFrame)-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        logger.info("Spliting the data")
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]

        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=self._test_size, random_state= self._random_state, stratify= y)

        validation_ratio = self._validation_size / (self._train_size + self._validation_size)

        X_train, X_validate, y_train, y_validate = train_test_split(X_train_val, y_train_val, test_size= validation_ratio, random_state= self._random_state, stratify= y_train_val)
        logger.info("Data Spliting Completed")
        return X_train, X_validate, X_test, y_train, y_validate, y_test