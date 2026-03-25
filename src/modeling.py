from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-maintenance-request")))

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import (
    ARTIFACTS_DIR,
    CONFUSION_MATRIX_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PREDICTIONS_PATH,
    SUMMARY_PATH,
)

matplotlib.use("Agg")


def build_models() -> dict[str, Pipeline]:
    common = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000)
    return {
        "logistic_regression": Pipeline(
            [
                ("tfidf", common),
                ("classifier", LogisticRegression(max_iter=1200, class_weight="balanced")),
            ]
        ),
        "linear_svc": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000)),
                ("classifier", LinearSVC(class_weight="balanced")),
            ]
        ),
        "multinomial_nb": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000)),
                ("classifier", MultinomialNB(alpha=0.5)),
            ]
        ),
    }


def train_and_evaluate(df: pd.DataFrame) -> dict:
    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(
        df["feature_text"],
        df["maintenance_group"],
        df["sample_id"],
        test_size=0.2,
        random_state=42,
        stratify=df["maintenance_group"],
    )

    metrics_rows: list[dict] = []
    predictions_frames: list[pd.DataFrame] = []
    trained_models: dict[str, Pipeline] = {}

    for model_name, pipeline in build_models().items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        trained_models[model_name] = pipeline

        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "macro_f1": f1_score(y_test, predictions, average="macro"),
                "weighted_f1": f1_score(y_test, predictions, average="weighted"),
            }
        )
        predictions_frames.append(
            pd.DataFrame(
                {
                    "sample_id": id_test.to_numpy(),
                    "actual_group": y_test.to_numpy(),
                    "predicted_group": predictions,
                    "model": model_name,
                }
            )
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)
    predictions_df = pd.concat(predictions_frames, ignore_index=True)

    best_model_name = metrics_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    best_predictions = predictions_df[predictions_df["model"] == best_model_name]["predicted_group"].to_numpy()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    joblib.dump(best_model, MODEL_PATH)

    labels = sorted(df["maintenance_group"].unique().tolist())
    matrix = confusion_matrix(y_test, best_predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=ax, xticks_rotation=35, colorbar=False)
    ax.set_title(f"Confusion Matrix - {best_model_name}")
    fig.tight_layout()
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = classification_report(y_test, best_predictions, output_dict=True)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else {}
    summary.update(
        {
            "best_model": best_model_name,
            "best_macro_f1": round(float(metrics_df.iloc[0]["macro_f1"]), 4),
            "best_accuracy": round(float(metrics_df.iloc[0]["accuracy"]), 4),
            "validation_size": int(len(y_test)),
            "labels": labels,
            "classification_report": report,
        }
    )
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_best_model() -> Pipeline:
    return joblib.load(MODEL_PATH)


def predict_request(model: Pipeline, feature_text: str) -> str:
    return model.predict([feature_text])[0]
