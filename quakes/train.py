
"""Part 2, Task D: train, evaluate, and save earthquake models."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from . import config
from .evaluate import (
    best_threshold,
    classification_report_dict,
    regression_report,
)
from .models import make_classifier, make_regressor

log = logging.getLogger("quakes.train")


def load_data(db_path=None):
    """Load the ETL output table from SQLite."""
    import sqlite3

    path = Path(db_path) if db_path is not None else (
        config.PROCESSED_DIR / "quakes.db"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Database not found: {path}. Run the ETL first."
        )

    with sqlite3.connect(str(path)) as connection:
        df = pd.read_sql_query("SELECT * FROM quakes", connection)

    if df.empty:
        raise ValueError("The quakes table contains no records")

    required = {"mag", "big_quake"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def main() -> None:
    """Train a model, compare it with a baseline, and save results."""
    parser = argparse.ArgumentParser(
        description="Train earthquake regression or classification models"
    )
    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=["dummy", "linear", "ridge", "poly", "logistic"],
        default=None,
    )
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    df = load_data(args.db_path)

    # Choose target and remove both targets from the feature matrix.
    target = "mag" if args.task == "regression" else "big_quake"
    X = df.drop(columns=["mag", "big_quake"], errors="ignore")
    y = df[target]

    if args.task == "classification" and y.nunique() < 2:
        raise ValueError(
            "Classification requires examples of both big and small quakes."
        )

    # Keep a separate test set until the final evaluation.
    stratify = y if args.task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=config.SEED,
        stratify=stratify,
    )

    if args.task == "regression":
        model_kind = args.model or "linear"
        if model_kind == "logistic":
            parser.error("--model logistic is only for classification")

        model = make_regressor(
            kind=model_kind,
            degree=args.degree,
            alpha=args.alpha,
        )
        baseline = make_regressor(kind="dummy")
        scoring = {
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        }
    else:
        model_kind = args.model or "logistic"
        if model_kind not in {"dummy", "logistic"}:
            parser.error(
                "Classification supports --model dummy or --model logistic"
            )

        model = make_classifier(
            kind=model_kind,
            class_weight="balanced" if args.balanced else None,
        )
        baseline = make_classifier(kind="dummy")
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc",
        }

    # Fit and evaluate the baseline using the same split.
    baseline.fit(X_train, y_train)
    if args.task == "regression":
        baseline_report = regression_report(
            y_test, baseline.predict(X_test)
        )
    else:
        baseline_proba = baseline.predict_proba(X_test)[:, 1]
        baseline_report = classification_report_dict(
            y_test,
            baseline.predict(X_test),
            baseline_proba,
        )

    # Cross-validation uses training data only.
    class_counts = y_train.value_counts() if args.task == "classification" else None
    cv_folds = 5
    if class_counts is not None:
        cv_folds = min(cv_folds, int(class_counts.min()))
        if cv_folds < 2:
            raise ValueError(
                "Not enough examples per class for cross-validation."
            )

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv_folds,
        scoring=scoring,
        return_train_score=False,
    )

    cv_report = {
        metric.removeprefix("test_"): float(values.mean())
        for metric, values in cv_results.items()
        if metric.startswith("test_")
    }

    threshold = None

    if args.task == "classification":
        # Generate out-of-fold probabilities using training data only.
        oof_proba = cross_val_predict(
            clone(model),
            X_train,
            y_train,
            cv=cv_folds,
            method="predict_proba",
        )[:, 1]

        threshold = best_threshold(y_train, oof_proba)

        model.fit(X_train, y_train)
        test_proba = model.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= threshold).astype(int)

        model_report = classification_report_dict(
            y_test, test_pred, test_proba
        )
        model_report["threshold"] = float(threshold)

    else:
        model.fit(X_train, y_train)
        model_report = regression_report(
            y_test, model.predict(X_test)
        )

    # Save the fitted pipeline and its JSON report.
    models_dir = config.ROOT / "models"
    reports_dir = config.ROOT / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{args.task}_{model_kind}.joblib"
    report_path = reports_dir / f"{args.task}_{model_kind}.json"

    import joblib
    joblib.dump(model, model_path)

    report = {
        "task": args.task,
        "model": model_kind,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "baseline_test": baseline_report,
        "cross_validation_train": cv_report,
        "model_test": model_report,
        "threshold": threshold,
        "model_path": str(model_path),
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=True)

    log.info("Baseline test metrics: %s", baseline_report)
    log.info("Training cross-validation metrics: %s", cv_report)
    log.info("Model test metrics: %s", model_report)
    log.info("Saved model to %s", model_path)
    log.info("Saved report to %s", report_path)


if __name__ == "__main__":
    main()
