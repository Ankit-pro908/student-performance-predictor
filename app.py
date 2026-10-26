"""
app.py
======
Intelligent Student Performance Prediction System - Streamlit front end.

Run:
    streamlit run app.py

Pages
-----
1. Predict        - four indicators in, band + confidence + action plan out
2. Model insights - accuracy, model comparison, confusion matrix, the tree
3. Dataset        - distributions, profiles, correlations, raw rows

All styling and HTML components live in ui.py so this file stays about
behaviour: loading the model, handling input, and laying out the page.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import predictor as P
import ui

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_data.csv"
MODEL_DIR = BASE_DIR / "model"
METRICS_PATH = MODEL_DIR / "metrics.json"
TREE_IMAGE_PATH = MODEL_DIR / "decision_tree.png"
TREE_RULES_PATH = MODEL_DIR / "decision_tree_rules.txt"

# Defaults for the four inputs (kept in session state so presets can change them).
DEFAULT_INPUTS = {
    "in_attendance": 80,
    "in_assignment": 70,
    "in_internal": 65,
    "in_study": 3.0,
}

# Demo profiles - close to the average profile of each band in the dataset.
PRESETS = {
    "excellent": {"attendance": 95, "assignment": 92, "internal": 93, "study": 5.5},
    "good": {"attendance": 85, "assignment": 80, "internal": 76, "study": 4.5},
    "average": {"attendance": 76, "assignment": 66, "internal": 60, "study": 2.7},
    "risk": {"attendance": 62, "assignment": 45, "internal": 38, "study": 1.0},
}

BAND_MEANING = {
    "Excellent": "On track for a strong final grade. The priority now is protecting the routine that produced this.",
    "Good": "Comfortably above average and one band away from Excellent.",
    "Average": "Sitting in the middle band - a few targeted fixes move this up quickly.",
    "Poor": "At risk of missing the expected outcome. Act within the next two weeks, not after the next exam.",
}

CONFIDENCE_NOTE = [
    (0.70, "High confidence - the profile clearly matches this band."),
    (0.50, "Moderate confidence - some indicators point to a neighbouring band."),
    (0.00, "Borderline - this student sits between two bands, so treat the advice as the priority."),
]

# Weights behind the transparent readiness score (see generate_dataset.py).
WEIGHTS = {
    "attendance_pct": 0.20,
    "assignment_pct": 0.25,
    "internal_pct": 0.35,
    "study_hours": 0.20,
}

FEATURE_TONE = {
    "attendance_pct": "cyan",
    "assignment_pct": "brand",
    "internal_pct": "violet",
    "study_hours": "amber",
}

PRIORITY_TONE = {"High": "red", "Medium": "amber", "Low": "brand"}

STATUS_SUB = {
    "poor": "currently below the recommended level",
    "average": "close, but still short of the target",
}

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_bundle() -> dict:
    return P.load_bundle()


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------
def plotly(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def run_prediction() -> dict:
    result = P.predict_performance(
        get_bundle(),
        st.session_state["in_attendance"],
        st.session_state["in_assignment"],
        st.session_state["in_internal"],
        st.session_state["in_study"],
    )
    st.session_state["result"] = result
    return result


def apply_preset(name: str) -> None:
    if not P.MODEL_PATH.exists():
        return
    preset = PRESETS[name]
    st.session_state["in_attendance"] = int(preset["attendance"])
    st.session_state["in_assignment"] = int(preset["assignment"])
    st.session_state["in_internal"] = int(preset["internal"])
    st.session_state["in_study"] = float(preset["study"])
    st.session_state["result_name"] = ""
    st.session_state["result_roll"] = ""
    run_prediction()


def apply_random() -> None:
    if not P.MODEL_PATH.exists():
        return
    row = get_dataset().sample(n=1, random_state=None).iloc[0]
    st.session_state["in_attendance"] = int(round(row["attendance_pct"]))
    st.session_state["in_assignment"] = int(round(row["assignment_pct"]))
    st.session_state["in_internal"] = int(round(row["internal_pct"]))
    st.session_state["in_study"] = float(round(row["study_hours"] * 2) / 2)
    st.session_state["result_name"] = str(row["student_id"])
    st.session_state["result_roll"] = ""
    run_prediction()


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def radar_chart(student: dict, average: dict) -> go.Figure:
    categories = ["Attendance", "Assignments", "Internals", "Study effort"]
    keys = ["attendance_pct", "assignment_pct", "internal_pct", "study_hours"]

    def to_scale(key: str, value: float) -> float:
        return min(value / 6 * 100, 100.0) if key == "study_hours" else value

    student_values = [round(to_scale(k, student[k]), 1) for k in keys]
    average_values = [round(to_scale(k, average[k]), 1) for k in keys]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=average_values,
            theta=categories,
            fill="toself",
            name="Cohort average",
            line=dict(color="#94a3b8", width=1.6, dash="dot"),
            fillcolor="rgba(148,163,184,0.16)",
            hovertemplate="%{theta}: %{r:.0f}/100<extra>Cohort average</extra>",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=student_values,
            theta=categories,
            fill="toself",
            name="This student",
            line=dict(color="#4f46e5", width=2.4),
            fillcolor="rgba(79,70,229,0.20)",
            hovertemplate="%{theta}: %{r:.0f}/100<extra>This student</extra>",
        )
    )
    ui.style_fig(fig, title="Student vs cohort average", height=336, legend=True)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=9, color="#a3adc0"),
                gridcolor="#e9eef8",
                linecolor="#e9eef8",
                tickvals=[25, 50, 75, 100],
            ),
            angularaxis=dict(
                gridcolor="#e9eef8",
                linecolor="#dfe6f2",
                tickfont=dict(size=11, color="#475569"),
            ),
        ),
        margin=dict(l=44, r=44, t=54, b=34),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.10, font=dict(size=10.5)),
    )
    return fig


def horizontal_bar(
    items: list[tuple[str, float]],
    colors: list[str],
    suffix: str = "%",
    title: str = "",
) -> go.Figure:
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{value:.1f}{suffix}" for value in values],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=11, color="#64748b"),
            hovertemplate="%{y}: %{x:.1f}" + suffix + "<extra></extra>",
        )
    )
    ui.style_fig(fig, title=title, height=max(210, 46 * len(items) + 62))
    ceiling = max(values + [1.0]) * 1.26
    fig.update_layout(
        bargap=0.44,
        xaxis=dict(
            range=[0, ceiling],
            ticksuffix=suffix,
            showgrid=True,
            gridcolor="#f1f4fa",
            showline=False,
        ),
        yaxis=dict(autorange="reversed", showgrid=False, automargin=True),
        margin=dict(l=8, r=30, t=56 if title else 20, b=10),
    )
    return fig


def class_distribution_chart(counts: pd.Series) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=list(counts.index),
            y=list(counts.values),
            marker=dict(
                color=[P.CLASS_STYLE[c]["color"] for c in counts.index],
                line=dict(width=0),
            ),
            text=[f"{value} · {value / counts.sum():.0%}" for value in counts.values],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=11, color="#64748b"),
            hovertemplate="%{x}: %{y} students<extra></extra>",
        )
    )
    ui.style_fig(fig, title="Class distribution", height=306)
    fig.update_layout(
        bargap=0.48,
        yaxis=dict(showgrid=True, gridcolor="#f1f4fa", range=[0, max(counts.values) * 1.22]),
        xaxis=dict(showgrid=False),
        margin=dict(l=22, r=18, t=56, b=12),
    )
    return fig


def correlation_chart(corr: pd.DataFrame) -> go.Figure:
    names = ["Attendance", "Assignments", "Internals", "Study hours"]
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=names,
            y=names,
            colorscale=[
                [0.0, "#f8fafc"],
                [0.25, "#e0e7ff"],
                [0.5, "#c7d2fe"],
                [0.75, "#818cf8"],
                [1.0, "#4338ca"],
            ],
            zmin=0,
            zmax=1,
            text=corr.values,
            texttemplate="%{text:.2f}",
            textfont=dict(size=12),
            showscale=False,
            xgap=4,
            ygap=4,
            hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
        )
    )
    ui.style_fig(fig, title="Correlation between indicators", height=330)
    fig.update_layout(
        yaxis=dict(autorange="reversed", showgrid=False),
        xaxis=dict(showgrid=False, side="bottom"),
        margin=dict(l=14, r=14, t=56, b=14),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(metrics: dict | None) -> None:
    with st.sidebar:
        ui.sidebar_brand("Performance Predictor", "Decision Tree · 4 indicators")

        ui.h('<div class="sb-label">Quick demo profiles</div>')
        col_a, col_b = st.columns(2)
        with col_a:
            st.button("🌟 Excellent", on_click=apply_preset, args=("excellent",), width="stretch")
            st.button("😐 Average", on_click=apply_preset, args=("average",), width="stretch")
        with col_b:
            st.button("🙂 Good", on_click=apply_preset, args=("good",), width="stretch")
            st.button("⚠️ At risk", on_click=apply_preset, args=("risk",), width="stretch")
        st.button("🎲 Random student", on_click=apply_random, width="stretch")
        st.caption("A profile fills the form and predicts immediately.")

        st.divider()
        ui.sidebar_steps(
            "How it works",
            [
                "Enter attendance, assignment marks, internal marks and daily study hours.",
                "The trained Decision Tree applies thresholds learned from 1,500 students.",
                "The student lands in one of four bands with a confidence score.",
                "The weakest indicators become a ranked, 30-day action plan.",
            ],
        )

        if metrics:
            st.divider()
            ui.sidebar_card(
                "Model card",
                [
                    ("Algorithm", "Decision Tree"),
                    ("Max depth", f"{metrics['decision_tree']['depth']} ({metrics['decision_tree']['leaves']} leaves)"),
                    ("Test accuracy", f"{metrics['holdout']['accuracy']:.1%}"),
                    ("Training rows", f"{metrics['dataset']['n_samples']:,}"),
                    ("Classes", f"{len(metrics['classes'])} bands"),
                ],
            )
        st.caption("Academic project demo · predictions are advisory, not a grade.")


# ---------------------------------------------------------------------------
# Predict tab
# ---------------------------------------------------------------------------
def render_form() -> None:
    with st.form("student_form", clear_on_submit=False):
        ui.h(
            '<div style="display:flex;align-items:center;gap:11px;margin:2px 0 10px">'
            '<div class="section-icon" style="margin:0">📝</div>'
            "<div>"
            '<div class="panel-title">Student profile</div>'
            '<div class="panel-sub">Four indicators, one prediction</div>'
            "</div>"
            "</div>"
        )

        col_name, col_roll = st.columns(2)
        with col_name:
            name = st.text_input("Student name", placeholder="e.g. Ananya Sharma", key="in_name")
        with col_roll:
            roll = st.text_input("Roll number", placeholder="e.g. 22CS1043", key="in_roll")

        ui.h('<div class="sb-label" style="margin:10px 0 2px">Academic indicators</div>')

        st.slider(
            "Attendance (%)",
            0,
            100,
            step=1,
            key="in_attendance",
            help="Percentage of classes attended this semester. 75% is the usual requirement.",
        )
        st.slider(
            "Assignment marks (%)",
            0,
            100,
            step=1,
            key="in_assignment",
            help="Average score across submitted assignments, as a percentage.",
        )
        st.slider(
            "Internal marks (%)",
            0,
            100,
            step=1,
            key="in_internal",
            help="Average of internal tests / mid-semester exams, as a percentage.",
        )
        st.slider(
            "Study hours per day",
            0.0,
            10.0,
            step=0.5,
            key="in_study",
            help="Self-study hours on a typical day, outside scheduled classes.",
        )

        st.caption("Marks are percentages - for example 18 out of 25 is 72%.")
        submitted = st.form_submit_button("🔮 Predict performance", width="stretch")

        if submitted:
            run_prediction()
            st.session_state["result_name"] = name.strip() or "Student"
            st.session_state["result_roll"] = roll.strip()


def render_empty_state() -> None:
    ui.empty_state(
        "🎯",
        "No prediction yet",
        "Set the four indicators and press Predict performance. "
        "Or load a demo profile from the sidebar to see the system working immediately.",
        ["Attendance", "Assignments", "Internals", "Study hours"],
    )
    ui.h(
        ui.callout(
            "🌳",
            "Every prediction is explainable",
            "The model is a shallow Decision Tree, so the exact if/else rules behind any band can be "
            "shown to a student or a teacher - no black box.",
            "brand",
        )
    )


def render_result(result: dict) -> None:
    label = result["label"]
    style = P.CLASS_STYLE[label]
    name = st.session_state.get("result_name") or "Student"
    roll = st.session_state.get("result_roll") or ""
    who = ui.esc(name) + (f" · {ui.esc(roll)}" if roll else "")

    chip_items = [
        {
            "label": item["label"],
            "value": item["display"],
            "color": item["color"],
            "icon": item["icon"],
        }
        for item in P.feature_status(result["inputs"])
    ]

    ring = ui.score_ring(result["score"], style["color"], style["soft"], "readiness")
    accent = ui.blend(style["color"], "#c7d2fe", 0.35)

    ui.h(
        f'<div class="result-card" style="--band:{style["color"]};--band-2:{accent}">'
        '<div class="result-grid">'
        '<div class="result-copy">'
        '<div class="result-eyebrow">Predicted performance band</div>'
        f'<div class="result-band"><span class="emoji">{style["emoji"]}</span>{ui.esc(label)}</div>'
        '<div class="result-conf">Model confidence '
        f'<b style="color:{style["color"]}">{result["confidence"]:.1%}</b>'
        '<span class="sep">·</span>readiness '
        f'<b>{result["score"]:.1f}/100</b></div>'
        f'<p class="result-sub">{ui.esc(BAND_MEANING[label])}</p>'
        f'{ui.chips(chip_items)}'
        f'<div class="result-who">Prediction for <b>{who}</b></div>'
        "</div>"
        f"{ring}"
        "</div>"
        "</div>"
    )

    probabilities = result["probabilities"]
    ui.prob_bars(
        [
            (band, probabilities[band] * 100, P.CLASS_STYLE[band]["color"])
            for band in P.CLASS_ORDER
        ],
        label,
        style["color"],
    )

    for threshold, text in CONFIDENCE_NOTE:
        if result["confidence"] >= threshold:
            ui.h(ui.softnote("ℹ️", ui.esc(text)))
            break

    if result["inputs"]["attendance_pct"] < 75:
        ui.h(
            ui.callout(
                "🚪",
                "Attendance is below the 75% requirement",
                "Students in that position are usually detained, so the training data caps their band at "
                "<b>Average</b> regardless of marks. Improving attendance is the fastest way out of that limit.",
                "amber",
                warn=True,
            )
        )


def render_decision_path(result: dict) -> None:
    with st.expander("🔎 Why this prediction? The tree path behind it"):
        steps = P.decision_path(get_bundle(), result["inputs"])
        ui.h(
            '<ol class="steps" style="margin-top:4px">'
            + "".join(f"<li>{ui.esc(step)}</li>" for step in steps)
            + "</ol>"
        )
        st.caption(
            "Each line is a real if/else rule the Decision Tree learned during training. "
            "That is what keeps the model explainable."
        )


def render_plan(result: dict) -> None:
    label = result["label"]
    values = result["inputs"]
    suggestions = P.generate_suggestions(values, label)
    gaps = P.biggest_gaps(values, limit=2)

    ui.section("Personalised improvement plan", "🛠️", f"{len(suggestions)} actions")

    if gaps:
        first = " and ".join(gaps)
        detail = (
            "Fix these two first. They have the largest gap to a healthy level, and both are directly "
            "in the student's control."
            if len(gaps) > 1
            else "Fix this first. It has the largest gap to a healthy level and is directly in the "
            "student's control."
        )
        ui.h(ui.callout("🎯", f"Start here: {first}", ui.esc(detail), "brand"))
    else:
        ui.h(
            ui.callout(
                "✅",
                "All four indicators are in a healthy range",
                "No urgent fixes are needed. The plan below is about protecting the routine.",
                "green",
            )
        )

    cols = st.columns(2, gap="large")
    for index, item in enumerate(suggestions):
        sub = item.get("sub")
        if not sub:
            sub = STATUS_SUB.get(item.get("status", ""), "band-level advice")
        card = ui.tip_card(
            {
                "icon": item["icon"],
                "area": item["area"],
                "badge": f"{item['priority']} priority",
                "tone": PRIORITY_TONE.get(item["priority"], "brand"),
                "sub": sub.replace("currently ", "").capitalize(),
                "message": item["message"],
                "action": item["action"],
            }
        )
        with cols[index % 2]:
            ui.h(card)


def render_profile(result: dict) -> None:
    dataset = get_dataset()
    average = {key: float(dataset[key].mean()) for key in P.FEATURES}
    values = result["inputs"]

    ui.section("Profile check", "🕸️", "against the 1,500-student cohort")

    col_chart, col_side = st.columns([1.06, 0.94], gap="large")

    with col_chart:
        plotly(radar_chart(values, average))
        st.caption("Study effort is rescaled: 6 hours a day counts as 100/100.")

    with col_side:
        strengths = P.strengths(values)
        if strengths:
            rows = ui.rowlist(
                [
                    {
                        "icon": item["icon"],
                        "label": item["label"],
                        "value": item["display"],
                        "status_label": item["status_label"],
                        "color": P.STATUS_COLOR[item["status"]],
                    }
                    for item in strengths
                ]
            )
        else:
            rows = ui.callout(
                "🧭",
                "No indicator is healthy yet",
                "All four inputs are below their recommended level, so the whole routine needs a reset "
                "rather than one fix. Start with attendance - it is the easiest to control.",
                "amber",
                warn=True,
            )

        score = max(result["score"], 0.1)
        meters = []
        for key in P.FEATURES:
            meta = P.FEATURE_META[key]
            raw = float(values[key])
            scaled = min(raw / 6, 1) * 100 if key == "study_hours" else raw
            weight = WEIGHTS[key] * 100
            contribution = weight * scaled / 100
            meters.append(
                {
                    "label": meta["short"],
                    "share": contribution / score * 100,
                    "value": f"{contribution:.0f} pts",
                    "tone": FEATURE_TONE[key],
                }
            )

        ui.h(
            '<div class="panel">'
            '<div class="panel-head"><div>'
            '<div class="panel-title">What is already working</div>'
            '<div class="panel-sub">Indicators the model reads as an asset</div>'
            "</div></div>"
            f"{rows}"
            "</div>"
        )
        ui.h('<div style="height:12px"></div>')
        ui.h(
            '<div class="panel">'
            '<div class="panel-head"><div>'
            '<div class="panel-title">Where the readiness score comes from</div>'
            f'<div class="panel-sub">Fixed project weights · total {result["score"]:.1f}/100</div>'
            "</div>"
            '<span class="panel-note">weight × score</span>'
            "</div>"
            f"{ui.weight_meter(meters)}"
            '<div class="panel-sub" style="margin-top:12px">The gauge is a transparent weighted average, '
            "kept separate from the tree. The model learns its own thresholds from the data.</div>"
            "</div>"
        )


# ---------------------------------------------------------------------------
# Insights tab
# ---------------------------------------------------------------------------
def render_insights(metrics: dict) -> None:
    ui.section("Model scorecard", "📊", "hold-out test set · 300 students")

    ui.tiles(
        [
            {
                "label": "Test accuracy",
                "value": f"{metrics['holdout']['accuracy']:.1%}",
                "hint": "Share of held-out students classified correctly",
                "icon": "🎯",
                "tone": "green",
            },
            {
                "label": "Macro F1",
                "value": f"{metrics['holdout']['macro_f1']:.3f}",
                "hint": "Every band weighted equally, not just the common ones",
                "icon": "⚖️",
                "tone": "brand",
            },
            {
                "label": "5-fold CV",
                "value": f"{metrics['cross_validation']['mean']:.1%}",
                "hint": f"Mean accuracy across five splits (± {metrics['cross_validation']['std']:.1%})",
                "icon": "🔁",
                "tone": "violet",
            },
            {
                "label": "Tree size",
                "value": f"{metrics['decision_tree']['leaves']} leaves",
                "hint": f"Depth {metrics['decision_tree']['depth']} · still readable by hand",
                "icon": "🌳",
                "tone": "cyan",
            },
        ]
    )

    ui.section("Model comparison and drivers", "🔍")

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        comparison = metrics["model_comparison"]
        labels = [row["model"] for row in comparison]
        values = [row["test_accuracy"] * 100 for row in comparison]
        colors = [ui.BRAND if label == "Decision Tree" else "#cbd5e1" for label in labels]
        plotly(horizontal_bar(list(zip(labels, values)), colors, "%", "Test accuracy by model"))
        st.caption(
            "The Decision Tree gives up about 3 points to the best model, but every prediction can be "
            "read as a set of if/else rules - which is what makes the feedback trustworthy."
        )

    with col_right:
        importance = metrics["feature_importance"]
        items = [(row["label"], row["importance"] * 100) for row in importance]
        colors = ["#4f46e5", "#6366f1", "#818cf8", "#a5b4fc"]
        plotly(horizontal_bar(items, colors[: len(items)], "%", "What drives the prediction"))
        st.caption(
            "Attendance matters twice: it shapes the score directly and it caps the band at Average "
            "below 75%, because students who fall short of the requirement are normally detained."
        )

    ui.section("Where the model gets confused", "🧩", "adjacent bands overlap by design")

    col_cm, col_report = st.columns([1.06, 0.94], gap="large")
    with col_cm:
        confusion = metrics["confusion_matrix"]
        ui.confusion_grid(
            confusion["matrix"],
            confusion["labels"],
            sum(sum(row) for row in confusion["matrix"]),
        )
        st.caption("Rows are the true band, columns are what the model predicted.")

    with col_report:
        report = {
            label: row
            for label, row in metrics["classification_report"].items()
            if isinstance(row, dict)
        }
        order = [c for c in P.CLASS_ORDER if c in report] + [
            key for key in ("macro avg", "weighted avg") if key in report
        ]
        rows = [
            [
                label,
                f"{report[label]['precision']:.3f}",
                f"{report[label]['recall']:.3f}",
                f"{report[label]['f1-score']:.3f}",
                f"{int(report[label]['support'])}",
            ]
            for label in order
        ]
        ui.table(
            ["Band", "Precision", "Recall", "F1", "Students"],
            rows,
            title="Per-band performance",
            subtitle="Precision = when it says this band, how often it is right",
            note="test set",
            numeric=[False, True, True, True, True],
            dot_colors={label: P.CLASS_STYLE[label]["color"] for label in P.CLASS_ORDER},
        )
        st.caption(
            "Adjacent-band mistakes (Good predicted as Average, for example) are far more common than "
            "distant ones, which is expected when labels come from a continuous score."
        )

    ui.section("The decision tree itself", "🌳", "explainability is the design goal")

    ui.h(
        ui.callout(
            "🧠",
            "Read it top-down",
            "Each box is a yes/no question and each coloured leaf is a predicted band. The tree is "
            "shallow enough that a teacher can follow any single prediction by hand.",
            "brand",
        )
    )
    if TREE_IMAGE_PATH.exists():
        st.image(str(TREE_IMAGE_PATH), width="stretch")
    else:
        st.info("Run `python train_model.py` to export the tree diagram.")

    if TREE_RULES_PATH.exists():
        with st.expander("View the learned rules as text"):
            st.code(TREE_RULES_PATH.read_text(encoding="utf-8"), language="text")


# ---------------------------------------------------------------------------
# Dataset tab
# ---------------------------------------------------------------------------
def render_dataset() -> None:
    df = get_dataset()
    detained = int((df["attendance_pct"] < 75).sum())

    ui.section("Training data at a glance", "🗂️", "synthetic cohort · seed 42")

    ui.tiles(
        [
            {
                "label": "Students",
                "value": f"{len(df):,}",
                "hint": "Rows the tree was trained and tested on",
                "icon": "👥",
                "tone": "brand",
            },
            {
                "label": "Indicators",
                "value": f"{len(P.FEATURES)}",
                "hint": "Attendance, assignments, internals, study time",
                "icon": "📌",
                "tone": "violet",
            },
            {
                "label": "Bands",
                "value": "4",
                "hint": "Excellent, Good, Average, Poor",
                "icon": "🏷️",
                "tone": "cyan",
            },
            {
                "label": "Below 75% attendance",
                "value": f"{detained:,}",
                "hint": "Capped at Average by the eligibility rule",
                "icon": "🚪",
                "tone": "amber",
            },
        ]
    )

    ui.section("Distribution and band profiles", "📈")

    counts = df["performance"].value_counts().reindex(P.CLASS_ORDER).fillna(0).astype(int)
    averages = (
        df.groupby("performance")[P.FEATURES + ["composite_score"]]
        .mean()
        .reindex(P.CLASS_ORDER)
        .round(1)
    )

    col_chart, col_table = st.columns([1, 1], gap="large")
    with col_chart:
        plotly(class_distribution_chart(counts))
        st.caption("The middle bands dominate - as they would in a real cohort.")

    with col_table:
        rows = [
            [
                band,
                f"{row['attendance_pct']:.1f}%",
                f"{row['assignment_pct']:.1f}%",
                f"{row['internal_pct']:.1f}%",
                f"{row['study_hours']:.1f} h",
                f"{row['composite_score']:.1f}",
            ]
            for band, row in averages.iterrows()
        ]
        ui.table(
            ["Band", "Attendance", "Assignments", "Internals", "Study", "Score"],
            rows,
            title="Average profile per band",
            subtitle="Mean indicator values for each performance band",
            numeric=[False, True, True, True, True, True],
            dot_colors={label: P.CLASS_STYLE[label]["color"] for label in P.CLASS_ORDER},
        )
        st.caption("Composite score is derived from the four inputs - reference only, never a feature.")

    ui.section("Summary statistics", "📐", "all four indicators")

    stats = df[P.FEATURES].describe().T
    rows = []
    for key, row in stats.iterrows():
        meta = P.FEATURE_META[key]
        suffix = " h" if key == "study_hours" else "%"
        rows.append(
            [meta["label"]]
            + [
                f"{row[column]:.1f}{suffix}"
                for column in ("mean", "std", "min", "25%", "50%", "75%", "max")
            ]
        )
    ui.table(
        ["Indicator", "Mean", "Std", "Min", "25%", "Median", "75%", "Max"],
        rows,
        title="Descriptive statistics",
        subtitle="Every value sits inside a plausible academic range",
        numeric=[False, True, True, True, True, True, True, True],
    )

    ui.section("Correlation between indicators", "🔗")

    corr = df[P.FEATURES].corr().round(2)
    col_heat, col_copy = st.columns([1.04, 0.96], gap="large")
    with col_heat:
        plotly(correlation_chart(corr))
    with col_copy:
        ui.h('<div style="height:6px"></div>')
        ui.h(
            ui.callout(
                "🤝",
                "The indicators move together",
                "A student who attends class usually also submits assignments and prepares for internals. "
                "That correlation is realistic - and it is why the tree can separate the bands so cleanly.",
                "brand",
            )
        )
        pairs = []
        for i, first in enumerate(P.FEATURES):
            for second in P.FEATURES[i + 1 :]:
                pairs.append(
                    {
                        "icon": "•",
                        "label": f"{P.FEATURE_META[first]['short']} ↔ {P.FEATURE_META[second]['short']}",
                        "value": f"{corr.loc[first, second]:.2f}",
                        "rank": float(corr.loc[first, second]),
                    }
                )
        pairs.sort(key=lambda item: item["rank"], reverse=True)
        ui.h(
            '<div class="panel">'
            '<div class="panel-head"><div>'
            '<div class="panel-title">Strongest relationships</div>'
            '<div class="panel-sub">Pearson correlation, 0 to 1</div>'
            "</div></div>"
            f"{ui.rowlist(pairs)}"
            "</div>"
        )

    ui.section("Raw records", "📋", "first 200 rows")

    st.caption(
        "composite_score is shown for reference only - it is derived from the four inputs and is never "
        "given to the model, because that would be data leakage. Students below 75% attendance are capped "
        "at Average, which is why some rows show a high score in the Average band."
    )
    st.dataframe(df.head(200), width="stretch", height=320)
    st.download_button(
        "⬇️ Download the dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="student_data.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ui.inject()

    metrics = get_metrics() if METRICS_PATH.exists() else None

    facts = [("Decision Tree", "explainable model"), ("4", "performance bands")]
    if metrics:
        facts = [
            (f"{metrics['holdout']['accuracy']:.1%}", "test accuracy"),
            ("1,500", "training students"),
            ("4", "performance bands"),
        ]

    ui.hero(
        "Machine learning · decision tree classification",
        "Intelligent student",
        "performance prediction",
        "Four academic indicators in, one clear band out - with the exact tree rules behind it and a "
        "prioritised improvement plan a student can act on this week.",
        facts,
    )

    for key, value in DEFAULT_INPUTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

    render_sidebar(metrics)

    if not P.MODEL_PATH.exists():
        st.error("The trained model was not found. Generate the data and train the model first:")
        st.code(
            "python generate_dataset.py\npython train_model.py\nstreamlit run app.py",
            language="bash",
        )
        st.stop()

    # Shareable demo links: app.py?demo=excellent|good|average|risk
    demo = st.query_params.get("demo")
    if demo in PRESETS and "result" not in st.session_state:
        apply_preset(demo)

    tab_predict, tab_insights, tab_data = st.tabs(
        ["🔮 Predict", "📊 Model insights", "🗂️ Dataset"]
    )

    with tab_predict:
        left, right = st.columns([0.88, 1.12], gap="large")
        with left:
            render_form()
            if "result" in st.session_state:
                render_decision_path(st.session_state["result"])
        with right:
            if "result" in st.session_state:
                render_result(st.session_state["result"])
            else:
                render_empty_state()

        if "result" in st.session_state:
            render_plan(st.session_state["result"])
            render_profile(st.session_state["result"])

    with tab_insights:
        if metrics:
            render_insights(metrics)
        else:
            st.info("Run `python train_model.py` to generate the metrics for this page.")

    with tab_data:
        if DATA_PATH.exists():
            render_dataset()
        else:
            st.info("Run `python generate_dataset.py` to create the dataset.")

    ui.footer(
        "<b>Intelligent Student Performance Prediction</b> · college mini-project · synthetic data",
        "Python · scikit-learn · Streamlit",
    )


if __name__ == "__main__":
    main()
