from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from .config import (
    DOWNLOAD_URL,
    GROUP_MAPPING,
    PROCESSED_DATA_PATH,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    RAW_DIR,
    SUMMARY_PATH,
)


TEXT_COLUMNS = ["descriptor", "location_type", "street_name", "borough", "agency", "status"]


def download_public_sample(refresh: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_PATH.exists() and not refresh:
        return RAW_DATA_PATH

    response = requests.get(DOWNLOAD_URL, timeout=60)
    response.raise_for_status()
    RAW_DATA_PATH.write_bytes(response.content)
    return RAW_DATA_PATH


def build_feature_text(frame: pd.DataFrame) -> pd.Series:
    values = frame[TEXT_COLUMNS].fillna("").astype(str)
    return values.agg(" | ".join, axis=1).str.lower()


def prepare_dataset(refresh_download: bool = False) -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        download_public_sample(refresh=refresh_download)

    df = pd.read_csv(RAW_DATA_PATH)
    df = df[df["complaint_type"].isin(GROUP_MAPPING)].copy()
    df["maintenance_group"] = df["complaint_type"].map(GROUP_MAPPING)
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["request_month"] = df["created_date"].dt.month.fillna(0).astype(int)
    df["request_hour"] = df["created_date"].dt.hour.fillna(0).astype(int)
    df["incident_zip"] = df["incident_zip"].fillna(0).astype(int).astype(str)
    df["feature_text"] = build_feature_text(df)
    df["sample_id"] = [f"REQ-{i:05d}" for i in range(1, len(df) + 1)]

    df = df[
        [
            "sample_id",
            "created_date",
            "agency",
            "complaint_type",
            "descriptor",
            "borough",
            "incident_zip",
            "latitude",
            "longitude",
            "location_type",
            "street_name",
            "status",
            "resolution_description",
            "request_month",
            "request_hour",
            "feature_text",
            "maintenance_group",
        ]
    ].copy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    summary = {
        "rows": int(len(df)),
        "complaint_types": int(df["complaint_type"].nunique()),
        "maintenance_groups": int(df["maintenance_group"].nunique()),
        "top_borough": str(df["borough"].mode().iat[0]),
        "top_group": str(df["maintenance_group"].mode().iat[0]),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return df
