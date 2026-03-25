from __future__ import annotations

from .data_pipeline import prepare_dataset
from .modeling import train_and_evaluate


def run_pipeline(refresh_download: bool = False) -> dict:
    df = prepare_dataset(refresh_download=refresh_download)
    return train_and_evaluate(df)
