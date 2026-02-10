from pathlib import Path
from typing import Tuple
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_raw(name: str) -> pd.DataFrame:
    path = RAW_DIR / name
    return pd.read_csv(path)


def save_processed(train: pd.DataFrame, val: pd.DataFrame, train_name: str = "train.csv", val_name: str = "val.csv") -> Tuple[Path, Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / train_name
    val_path = PROCESSED_DIR / val_name
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    return train_path, val_path
