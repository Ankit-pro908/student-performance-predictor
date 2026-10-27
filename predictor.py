"""
predictor.py
============
Shared inference logic for the Student Performance Prediction System.

Both the Streamlit app (app.py) and the console version (predict.py) import
from this file, so the prediction and the advice are always identical no
matter how the project is run.

Contents
--------
* load_bundle()          - load the trained Decision Tree from disk
* predict_performance()  - features in, label + probabilities + score out
* feature_status()       - rate each indicator against academic benchmarks
* generate_suggestions() - turn weak indicators into an action plan
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import pandas as pd

from generate_dataset import composite_score

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "student_performance_model.pkl"

# The four inputs the model was trained on - keep this order in sync with train_model.py.
FEATURES = ["attendance_pct", "assignment_pct", "internal_pct", "study_hours"]

CLASS_ORDER = ["Excellent", "Good", "Average", "Poor"]

CLASS_STYLE = {
    "Excellent": {"emoji": "🌟", "color": "#15803d", "soft": "#dcfce7"},
    "Good": {"emoji": "🙂", "color": "#1d4ed8", "soft": "#dbeafe"},
    "Average": {"emoji": "😐", "color": "#b45309", "soft": "#fef3c7"},
    "Poor": {"emoji": "⚠️", "color": "#b91c1c", "soft": "#fee2e2"},
}

# Academic benchmarks. The first three values are the cut-offs for
# "excellent", "good" and "average"; anything below the third is "poor".
# 75% attendance is the usual university requirement, so it is used as the line.
FEATURE_META = {
    "attendance_pct": {
        "label": "Attendance",
        "short": "Attendance",
        "icon": "📅",
        "unit": "%",
        "cuts": (90.0, 80.0, 75.0),
    },
    "assignment_pct": {
        "label": "Assignment marks",
        "short": "Assignments",
        "icon": "📝",
        "unit": "%",
        "cuts": (85.0, 70.0, 50.0),
    },
    "internal_pct": {
        "label": "Internal marks",
        "short": "Internals",
        "icon": "🧪",
        "unit": "%",
        "cuts": (85.0, 70.0, 50.0),
    },
    "study_hours": {
        "label": "Study hours",
        "short": "Study hours",
        "icon": "⏱️",
        "unit": " h/day",
        "cuts": (5.0, 3.5, 2.0),
    },
}

STATUS_LABEL = {
    "excellent": "Excellent",
    "good": "Good",
    "average": "Needs attention",
    "poor": "Critical",
}

STATUS_COLOR = {
    "excellent": "#15803d",
    "good": "#1d4ed8",
    "average": "#b45309",
    "poor": "#b91c1c",
}

PRIORITY_STYLE = {
    "High": {"color": "#b91c1c", "soft": "#fee2e2"},
    "Medium": {"color": "#b45309", "soft": "#fef3c7"},
    "Low": {"color": "#1d4ed8", "soft": "#dbeafe"},
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_bundle(path: Path | str = MODEL_PATH) -> dict:
    """Load the saved model bundle and fail with a helpful message if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}.\n"
            "Run these two commands first:\n"
            "    python generate_dataset.py\n"
            "    python train_model.py"
        )
    with warnings.catch_warnings():
        # A model pickled by an older scikit-learn still loads and predicts
        # correctly. Streamlit Community Cloud installs the newest release that
        # has a wheel for its Python version, so the patch release there can
        # differ from the one used to train - that warning is just noise.
        warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_performance(
    bundle: dict,
    attendance_pct: float,
    assignment_pct: float,
    internal_pct: float,
    study_hours: float,
) -> dict:
    """Predict the performance band and return everything the UI needs."""
    values = [attendance_pct, assignment_pct, internal_pct, study_hours]
    row = pd.DataFrame([values], columns=FEATURES)

    model = bundle["model"]
    label = model.predict(row)[0]

    proba = model.predict_proba(row)[0]
    by_class = {cls: float(p) for cls, p in zip(model.classes_, proba)}
    probabilities = {cls: by_class.get(cls, 0.0) for cls in CLASS_ORDER}

    # A transparent weighted score, kept separate from the model output.
    # It is only used for the "overall readiness" gauge, never for the label.
    score = float(composite_score(*values))

    return {
        "label": label,
        "confidence": float(max(probabilities.values())),
        "probabilities": probabilities,
        "score": round(score, 1),
        "inputs": dict(zip(FEATURES, values)),
    }


# ---------------------------------------------------------------------------
# Rating and advice
# ---------------------------------------------------------------------------
def _status(value: float, cuts: tuple[float, float, float]) -> str:
    excellent, good, average = cuts
    if value >= excellent:
        return "excellent"
    if value >= good:
        return "good"
    if value >= average:
        return "average"
    return "poor"


def feature_status(values: dict) -> list[dict]:
    """Rate each indicator and describe it for the UI."""
    out = []
    for key in FEATURES:
        meta = FEATURE_META[key]
        value = float(values[key])
        status = _status(value, meta["cuts"])
        out.append(
            {
                "key": key,
                "label": meta["label"],
                "icon": meta["icon"],
                "value": value,
                "display": format_value(key, value),
                "status": status,
                "status_label": STATUS_LABEL[status],
                "color": STATUS_COLOR[status],
            }
        )
    return out


def format_value(key: str, value: float) -> str:
    meta = FEATURE_META[key]
    if key == "study_hours":
        return f"{value:.1f}{meta['unit']}"
    return f"{value:.0f}{meta['unit']}"


# Message + action for every weak indicator.
# {value} and {target} are filled in at run time.
ADVICE = {
    ("attendance_pct", "poor"): {
        "message": (
            "Attendance is {value}, which is below the 75% requirement. Classes are where "
            "the syllabus is actually explained, so missed lectures quietly turn into lost "
            "internal marks and last-minute panic before exams."
        ),
        "action": (
            "Attend at least 90% of lectures and labs from next week. Sit in the front rows, "
            "and borrow notes immediately for anything you have already missed."
        ),
        "target": 90,
    },
    ("attendance_pct", "average"): {
        "message": (
            "Attendance is {value} - only just above the minimum. There is almost no margin "
            "left for illness, competitions or family emergencies."
        ),
        "action": (
            "Push attendance towards {target}% so a few unavoidable absences do not put you "
            "below the eligibility limit. Track it every week instead of every month."
        ),
        "target": 90,
    },
    ("assignment_pct", "poor"): {
        "message": (
            "Assignment average is {value}. Assignments are the easiest marks in the internal "
            "assessment and this is where you are losing the most ground."
        ),
        "action": (
            "Submit every assignment on time. Start three days before the deadline, use the "
            "rubric as a checklist, and get one draft reviewed before you submit."
        ),
        "target": 75,
    },
    ("assignment_pct", "average"): {
        "message": (
            "Assignments are averaging {value}. You are submitting regularly, but presentation "
            "and completeness are costing you marks you could easily keep."
        ),
        "action": (
            "Re-read the question before starting, add worked examples or diagrams where they "
            "help, and proofread once before submitting. Aim for {target}%."
        ),
        "target": 85,
    },
    ("internal_pct", "poor"): {
        "message": (
            "Internal exam marks are {value}. Internals carry roughly 30% weight and are the "
            "strongest signal of how the final exam will go."
        ),
        "action": (
            "Revise every unit within 24 hours of the lecture, solve the last three years' "
            "question papers under time pressure, and attend all revision classes."
        ),
        "target": 70,
    },
    ("internal_pct", "average"): {
        "message": (
            "Internal marks are {value}. The basics are there, but marks are being lost on "
            "application-based and long-answer questions."
        ),
        "action": (
            "Practise full-length answers with a timer and get them corrected. Spend extra time "
            "on the units where you scored the lowest. Target {target}%."
        ),
        "target": 80,
    },
    ("study_hours", "poor"): {
        "message": (
            "You are studying about {value}. That is not enough to keep pace with a full "
            "syllabus, so the backlog grows every single week."
        ),
        "action": (
            "Start with two 50-minute focused blocks per day, phone in another room. Build up to "
            "3+ hours within two weeks rather than trying a 6-hour day tomorrow."
        ),
        "target": 3.5,
    },
    ("study_hours", "average"): {
        "message": (
            "Study time of {value} is workable during teaching weeks, but leaves very "
            "little room for revision once exams begin."
        ),
        "action": (
            "Add one focused hour, ideally in the morning, and end each session with 15 minutes "
            "of active recall instead of re-reading. Aim for {target} h/day."
        ),
        "target": 5,
    },
}

# Extra advice attached to the predicted band.
BAND_ADVICE = {
    "Excellent": [
        {
            "area": "Consolidate",
            "icon": "🏆",
            "priority": "Low",
            "message": (
                "This profile is in the top band. Protect the routine that produced it - "
                "consistency, not intensity, is what keeps results here."
            ),
            "action": (
                "Keep the same weekly schedule through the semester, including the weeks when "
                "there is no exam nearby."
            ),
        },
        {
            "area": "Stretch further",
            "icon": "🚀",
            "priority": "Low",
            "message": (
                "Marks are no longer the limiting factor, so the next gain comes from depth and "
                "from how you use the knowledge."
            ),
            "action": (
                "Take on a project, a coding contest or a paper presentation in this subject, and "
                "mentor a classmate who is struggling - teaching exposes gaps fast."
            ),
        },
    ],
    "Good": [
        {
            "area": "Close the gap",
            "icon": "🎯",
            "priority": "Medium",
            "message": (
                "You are one band away from Excellent. The jump usually needs only 5-8 extra "
                "marks in internals plus consistently high-scoring assignments."
            ),
            "action": (
                "For the next 30 days, focus only on the weakest indicators listed above and "
                "measure them weekly."
            ),
        }
    ],
    "Average": [
        {
            "area": "Focus",
            "icon": "🧭",
            "priority": "Medium",
            "message": (
                "The profile sits in the middle band, which means there is clear room to move up "
                "without changing everything at once."
            ),
            "action": (
                "Fix the highest-priority area above completely before adding another. Trying to "
                "repair all four at the same time rarely survives week two."
            ),
        }
    ],
    "Poor": [
        {
            "area": "Get support now",
            "icon": "🆘",
            "priority": "High",
            "message": (
                "This profile is in the at-risk band. One weak semester is recoverable, but it "
                "needs action within the next two weeks, not after the next exam."
            ),
            "action": (
                "Book a slot with your faculty mentor or academic advisor this week, show them "
                "these numbers, and agree on a weekly check-in."
            ),
        }
    ],
}


def generate_suggestions(values: dict, label: str) -> list[dict]:
    """Build a prioritised action plan from the student's weakest indicators."""
    suggestions: list[dict] = []

    for key in FEATURES:
        meta = FEATURE_META[key]
        value = float(values[key])
        status = _status(value, meta["cuts"])
        if status not in ("poor", "average"):
            continue

        template = ADVICE[(key, status)]
        target = template["target"]
        suggestions.append(
            {
                "area": meta["label"],
                "icon": meta["icon"],
                "priority": "High" if status == "poor" else "Medium",
                "status": status,
                "message": template["message"].format(value=format_value(key, value)),
                "action": template["action"].format(target=target),
                # How far the student is from the recommended level (for sorting).
                "gap": max(0.0, target - value),
            }
        )

    # Biggest gaps first, and critical items before moderate ones.
    suggestions.sort(key=lambda s: (s["priority"] != "High", -s["gap"]))

    suggestions.extend(BAND_ADVICE.get(label, []))
    return suggestions


def decision_path(bundle: dict, values: dict) -> list[str]:
    """
    Walk the trained tree the same way the model did and return a readable
    list of the conditions that led to the predicted leaf.

    This is what makes a Decision Tree attractive for this project: the
    prediction is not a black box, it is a set of if/else tests that a
    teacher or student can check by hand.
    """
    model = bundle["model"]
    tree = model.tree_
    row = [float(values[f]) for f in FEATURES]

    node = 0
    steps: list[str] = []
    while tree.children_left[node] != -1:
        feature_index = tree.feature[node]
        threshold = tree.threshold[node]
        key = FEATURES[feature_index]
        meta = FEATURE_META[key]
        value = row[feature_index]

        if value <= threshold:
            branch_text = f"{meta['label']} ≤ {threshold:.1f}"
            node = tree.children_left[node]
        else:
            branch_text = f"{meta['label']} > {threshold:.1f}"
            node = tree.children_right[node]

        steps.append(
            f"{branch_text}{meta['unit']}   (your value: {format_value(key, value)})"
        )

    students_in_leaf = int(tree.weighted_n_node_samples[node])
    steps.append(f"→ reached leaf with {students_in_leaf} similar training students")
    return steps


def strengths(values: dict) -> list[dict]:
    """Indicators that are already in the good/excellent range."""
    out = []
    for key in FEATURES:
        meta = FEATURE_META[key]
        value = float(values[key])
        status = _status(value, meta["cuts"])
        if status in ("good", "excellent"):
            out.append(
                {
                    "label": meta["label"],
                    "icon": meta["icon"],
                    "display": format_value(key, value),
                    "status": status,
                    "status_label": STATUS_LABEL[status],
                }
            )
    return out


def biggest_gaps(values: dict, limit: int = 2) -> list[str]:
    """Short names of the indicators most worth fixing first."""
    scored = []
    for key in FEATURES:
        meta = FEATURE_META[key]
        value = float(values[key])
        status = _status(value, meta["cuts"])
        if status not in ("poor", "average"):
            continue
        target = ADVICE[(key, status)]["target"]
        scored.append((max(0.0, target - value), meta["short"]))
    scored.sort(reverse=True)
    return [name for _, name in scored[:limit]]
