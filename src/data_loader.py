from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, path: str | Path) -> None:
        self._zip_path = Path(path)
        self._output_directory = self._zip_path.parent
        if not self._zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {self._zip_path}")

    def save(self)-> None:
        logger.info("Saving the CSV file to destination")

        with ZipFile(self._zip_path, "r") as zip_file:
            csv_name = next(name for name in zip_file.namelist() if name.endswith(".csv"))
            zip_file.extractall(self._output_directory)
            csv_path = self._output_directory / csv_name
            if not csv_path.exists():
                zip_file.extractall(self._output_directory)
        logger.info("CSV filed saved to the destination")

    @staticmethod
    def load(csv_path: str | Path) -> pd.DataFrame:
        logger.info("Loading the CSV file from the destination")
        return pd.read_csv(csv_path)