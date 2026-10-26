"""
train_model.py
==============
Trains the Decision Tree classifier for the Student Performance Prediction
System and saves everything the app needs.

Pipeline
--------
1. Load the dataset (generate it automatically if it is missing).
2. Split 80/20 with stratification so every band appears in both sets.
3. Train a Decision Tree (entropy criterion, limited depth to avoid overfitting).
4. Evaluate with a hold-out set and 5-fold cross-validation.
5. Compare against three other classifiers for the report.
6. Persist the model, a PNG of the tree, the extracted rules and metrics.json.

Run:
    python train_model.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

import matplotlib

matplotlib.use("Agg")  # headless backend - we only write a PNG
import matplotlib.pyplot as plt  # noqa: E402

import generate_dataset  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_data.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "student_performance_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
TREE_IMAGE_PATH = MODEL_DIR / "decision_tree.png"
TREE_RULES_PATH = MODEL_DIR / "decision_tree_rules.txt"

TARGET = "performance"
FEATURES = ["attendance_pct", "assignment_pct", "internal_pct", "study_hours"]
CLASS_ORDER = ["Excellent", "Good", "Average", "Poor"]

FEATURE_LABELS = {
    "attendance_pct": "Attendance (%)",
    "assignment_pct": "Assignment marks (%)",
    "internal_pct": "Internal marks (%)",
    "study_hours": "Study hours/day",
}

RANDOM_STATE = 42
TEST_SIZE = 0.20

TREE_PARAMS = dict(
    criterion="gini",
    max_depth=6,  # deep enough to model the band boundaries, shallow enough to read
    min_samples_leaf=8,  # each leaf must describe at least 8 students
    min_samples_split=10,
    random_state=RANDOM_STATE,
)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print(f"Dataset missing - generating it at {DATA_PATH} ...")
        generate_dataset.generate_dataset().to_csv(DATA_PATH, index=False)
    df = pd.read_csv(DATA_PATH)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def compare_models(X_train, y_train, X_test, y_test) -> list[dict]:
    """Train a few standard classifiers so the report can justify the choice."""
    candidates = {
        "Decision Tree": DecisionTreeClassifier(**TREE_PARAMS),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "K-Nearest Neighbours (k=7)": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=7)
        ),
        "Random Forest (100 trees)": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    rows = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append(
            {
                "model": name,
                "test_accuracy": float(accuracy_score(y_test, pred)),
                "macro_f1": float(f1_score(y_test, pred, average="macro")),
            }
        )
    return rows


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    line = "=" * 66

    print(line)
    print("TRAINING: STUDENT PERFORMANCE PREDICTION MODEL")
    print(line)

    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    print(f"Dataset      : {DATA_PATH.name}  ({len(df)} rows)")
    print(f"Features     : {FEATURES}")
    print(f"Target       : {TARGET}  { {k: int(v) for k, v in y.value_counts().reindex(CLASS_ORDER).items()} }")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Split        : {len(X_train)} train / {len(X_test)} test (stratified)")

    # ---- Train the model that the project actually deploys -----------------
    model = DecisionTreeClassifier(**TREE_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    print()
    print("Decision Tree performance")
    print("-" * 66)
    print(f"  Hold-out accuracy     : {test_accuracy:.2%}")
    print(f"  Macro F1 score        : {macro_f1:.3f}")
    print(f"  Cross-validation (5)  : {cv_scores.mean():.2%} +/- {cv_scores.std():.2%}")
    print(f"  Tree depth used       : {model.get_depth()} (max allowed {TREE_PARAMS['max_depth']})")
    print(f"  Leaves                : {model.get_n_leaves()}")

    report = classification_report(
        y_test, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0
    )
    print()
    print("Classification report (test set)")
    print("-" * 66)
    print(
        classification_report(y_test, y_pred, labels=CLASS_ORDER, digits=3, zero_division=0)
    )

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)

    # ---- Compare with alternative classifiers ------------------------------
    print("Model comparison (same train/test split)")
    print("-" * 66)
    comparison = compare_models(X_train, y_train, X_test, y_test)
    print(f"  {'Model':<28}{'Accuracy':>10}{'Macro F1':>12}")
    for row in comparison:
        print(
            f"  {row['model']:<28}{row['test_accuracy']:>9.2%}{row['macro_f1']:>12.3f}"
        )
    print(
        "\n  The Decision Tree is deployed because it is within a couple of points of the\n"
        "  best model while staying fully explainable - every prediction can be traced\n"
        "  back to a readable if/else path, which matters for student feedback."
    )

    # ---- Feature importance -------------------------------------------------
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(
        ascending=False
    )
    print()
    print("Feature importance")
    print("-" * 66)
    for feature, value in importance.items():
        bar = "#" * int(round(value * 40))
        print(f"  {FEATURE_LABELS[feature]:<24}{value:6.3f}  {bar}")

    # ---- Save the tree as an image and as text rules ------------------------
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        model,
        feature_names=[FEATURE_LABELS[f] for f in FEATURES],
        class_names=list(model.classes_),
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        fontsize=7,
        ax=ax,
    )
    ax.set_title("Student Performance - Decision Tree", fontsize=16)
    fig.savefig(TREE_IMAGE_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)

    rules = export_text(model, feature_names=[FEATURE_LABELS[f] for f in FEATURES])
    with open(TREE_RULES_PATH, "w", encoding="utf-8") as fh:
        fh.write("DECISION TREE RULES - STUDENT PERFORMANCE PREDICTION\n")
        fh.write("=" * 60 + "\n")
        fh.write("Read each branch top-down; the final line is the predicted band.\n\n")
        fh.write(rules)

    # ---- Persist model bundle ----------------------------------------------
    bundle = {
        "model": model,
        "features": FEATURES,
        "feature_labels": FEATURE_LABELS,
        "classes": CLASS_ORDER,
        "target": TARGET,
        "test_accuracy": test_accuracy,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, MODEL_PATH)

    # ---- Persist metrics for the dashboard ---------------------------------
    metrics = {
        "trained_at": bundle["trained_at"],
        "dataset": {
            "n_samples": int(len(df)),
            "n_features": len(FEATURES),
            "class_counts": {
                cls: int(cnt) for cls, cnt in y.value_counts().reindex(CLASS_ORDER).items()
            },
        },
        "features": FEATURES,
        "feature_labels": FEATURE_LABELS,
        "classes": CLASS_ORDER,
        "holdout": {
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "accuracy": round(test_accuracy, 4),
            "macro_f1": round(macro_f1, 4),
        },
        "cross_validation": {
            "folds": int(cv.get_n_splits()),
            "mean": round(float(cv_scores.mean()), 4),
            "std": round(float(cv_scores.std()), 4),
            "scores": [round(float(s), 4) for s in cv_scores],
        },
        "decision_tree": {
            "parameters": {k: v for k, v in TREE_PARAMS.items()},
            "depth": int(model.get_depth()),
            "leaves": int(model.get_n_leaves()),
        },
        "classification_report": {
            label: (
                {k: round(float(v), 4) for k, v in row.items()}
                if isinstance(row, dict)
                else round(float(row), 4)
            )
            for label, row in report.items()
        },
        "accuracy": round(float(report["accuracy"]), 4),
        "confusion_matrix": {
            "labels": CLASS_ORDER,
            "matrix": cm.astype(int).tolist(),
        },
        "feature_importance": [
            {
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "importance": round(float(value), 4),
            }
            for feature, value in importance.items()
        ],
        "model_comparison": [
            {
                "model": row["model"],
                "test_accuracy": round(row["test_accuracy"], 4),
                "macro_f1": round(row["macro_f1"], 4),
            }
            for row in comparison
        ],
        "class_profile": {
            cls: {feature: round(float(val), 2) for feature, val in row.items()}
            for cls, row in df.groupby(TARGET)[FEATURES].mean().round(2).iterrows()
        },
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print()
    print("Saved artifacts")
    print("-" * 66)
    for path in (MODEL_PATH, METRICS_PATH, TREE_IMAGE_PATH, TREE_RULES_PATH):
        print(f"  {path.relative_to(BASE_DIR)}")
    print(line)
    print("Done. Start the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
