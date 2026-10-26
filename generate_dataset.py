"""
generate_dataset.py
===================
Creates the synthetic dataset used by the Student Performance Prediction System.

Why synthetic data?
-------------------
A real college cannot publish student records, so we simulate a realistic
population.  Each student is given two hidden traits - *diligence* and
*ability* - and the four academic indicators are generated from those traits
plus random noise.  The performance label is then derived from a weighted
composite score.

Because the indicators overlap between neighbouring classes, the problem is a
genuine classification task (not a trivially separable one).  Two students with
identical inputs can still end up in different bands, which is exactly the kind
of uncertainty a classifier has to model.

One domain rule is applied on top of the score: a student below 75% attendance
is normally detained, so the band is capped at "Average" regardless of marks.

Run:
    python generate_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_data.csv"

N_SAMPLES = 1500
SEED = 42

# Weights used to build the overall academic score (out of 100).
WEIGHTS = {
    "attendance_pct": 0.20,
    "assignment_pct": 0.25,
    "internal_pct": 0.35,
    "study_pct": 0.20,
}

# Lower bound of each performance band (checked from highest to lowest).
BANDS: list[tuple[str, float]] = [
    ("Excellent", 85.0),
    ("Good", 70.0),
    ("Average", 50.0),
    ("Poor", 0.0),
]

CLASS_ORDER = ["Excellent", "Good", "Average", "Poor"]

# Students below this attendance level are usually detained / not eligible to
# appear for the exam, so they cannot be placed in Good or Excellent no matter
# how well they score elsewhere. This is a real institutional rule, and it is
# what makes attendance a meaningful feature for the model.
ATTENDANCE_REQUIREMENT = 75.0


def study_to_pct(study_hours):
    """Convert daily study hours into a 0-100 score (6 h/day counts as full effort)."""
    return np.clip(np.asarray(study_hours, dtype=float) / 6.0, 0.0, 1.0) * 100.0


def composite_score(attendance, assignment, internal, study_hours):
    """Weighted overall academic score out of 100 (before the attendance rule)."""
    return (
        WEIGHTS["attendance_pct"] * np.asarray(attendance, dtype=float)
        + WEIGHTS["assignment_pct"] * np.asarray(assignment, dtype=float)
        + WEIGHTS["internal_pct"] * np.asarray(internal, dtype=float)
        + WEIGHTS["study_pct"] * study_to_pct(study_hours)
    )


def score_to_band(score: float) -> str:
    """Map a 0-100 score to one of the four performance bands."""
    for label, threshold in BANDS:
        if score >= threshold:
            return label
    return "Poor"


def apply_attendance_rule(bands, attendance):
    """
    Cap the band at 'Average' for students below the attendance requirement.

    Works on a list of labels and the matching attendance values, so it can be
    used for a single student or for a whole column.
    """
    capped = []
    for band, att in zip(bands, attendance):
        if float(att) < ATTENDANCE_REQUIREMENT and band in ("Excellent", "Good"):
            capped.append("Average")
        else:
            capped.append(band)
    return capped


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Build a synthetic student population and return it as a DataFrame."""
    rng = np.random.default_rng(seed)

    # Hidden traits shared by all indicators -> features are naturally correlated.
    diligence = rng.normal(0.0, 1.0, n_samples)
    ability = rng.normal(0.0, 1.0, n_samples)

    attendance = 80 + 6 * diligence + 3 * ability + rng.normal(0, 10, n_samples)
    assignment = 72 + 11 * diligence + 9 * ability + rng.normal(0, 7, n_samples)
    internal = 68 + 10 * diligence + 13 * ability + rng.normal(0, 8, n_samples)
    study_hours = 3.5 + 1.4 * diligence + 0.4 * ability + rng.normal(0, 1.0, n_samples)

    # Keep every value inside a plausible range.
    attendance = np.clip(attendance, 40, 100)
    assignment = np.clip(assignment, 10, 100)
    internal = np.clip(internal, 10, 100)
    study_hours = np.clip(study_hours, 0, 10)

    # Measurement noise: the same inputs can still produce a slightly different result.
    # 2.5 points of noise keeps the classes overlapping without making the
    # problem unpredictable (roughly 10% of points sit near a band boundary).
    score = composite_score(attendance, assignment, internal, study_hours)
    score = np.clip(score + rng.normal(0, 2.5, n_samples), 0, 100)

    df = pd.DataFrame(
        {
            "student_id": [f"STU{i:04d}" for i in range(1, n_samples + 1)],
            "attendance_pct": attendance.round(1),
            "assignment_pct": assignment.round(1),
            "internal_pct": internal.round(1),
            "study_hours": study_hours.round(1),
            "composite_score": score.round(1),
            "performance": apply_attendance_rule(
                [score_to_band(s) for s in score], attendance
            ),
        }
    )
    return df


def summarise(df: pd.DataFrame) -> None:
    """Print a short report about the generated dataset."""
    line = "=" * 62
    print(line)
    print("STUDENT DATASET GENERATED")
    print(line)
    print(f"Rows            : {len(df)}")
    print(f"Columns         : {list(df.columns)}")
    print(f"Missing values  : {int(df.isna().sum().sum())}")
    print()
    print("Class distribution")
    print("-" * 62)
    counts = df["performance"].value_counts().reindex(CLASS_ORDER)
    for label, count in counts.items():
        share = count / len(df) * 100
        bar = "#" * int(round(share / 2))
        print(f"  {label:<10} {count:>5}  ({share:5.1f}%)  {bar}")
    print()
    print("Average indicators per class")
    print("-" * 62)
    profile = df.groupby("performance")[
        ["attendance_pct", "assignment_pct", "internal_pct", "study_hours", "composite_score"]
    ].mean().reindex(CLASS_ORDER).round(1)
    print(profile.to_string())
    detained = int((df["attendance_pct"] < ATTENDANCE_REQUIREMENT).sum())
    print()
    print(f"Attendance rule : {detained} students below {ATTENDANCE_REQUIREMENT:.0f}% "
          "(band capped at Average)")
    print(line)


def main() -> None:
    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)
    summarise(df)
    print(f"Saved -> {DATA_PATH}")


if __name__ == "__main__":
    main()
