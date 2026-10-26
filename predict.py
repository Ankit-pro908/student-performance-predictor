"""
predict.py
==========
Console version of the Intelligent Student Performance Prediction System.

Useful when you want to demonstrate the model without opening a browser
(for example in a lab machine or during a viva).

Usage:
    python predict.py            # asks for the four indicators
    python predict.py --demo     # runs a fixed at-risk sample instantly
"""

from __future__ import annotations

import sys

import predictor as P

WIDTH = 66
RULE = "=" * WIDTH
THIN = "-" * WIDTH

DEMO_SAMPLE = {
    "attendance": 62.0,
    "assignment": 45.0,
    "internal": 38.0,
    "study": 1.0,
}


def setup_console() -> None:
    """Make sure emoji and the box characters print on any console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def bar(percentage: float, width: int = 22) -> str:
    filled = int(round(max(0.0, min(percentage, 100.0)) / 100 * width))
    return "#" * filled + "." * (width - filled)


def ask_number(label: str, minimum: float, maximum: float, default: float) -> float:
    """Prompt until the user gives a number inside the allowed range."""
    while True:
        raw = input(f"  {label:<26}[default {default:g}] : ").strip()
        if raw == "":
            return float(default)
        try:
            value = float(raw)
        except ValueError:
            print("    ! Please enter a number.")
            continue
        if not minimum <= value <= maximum:
            print(f"    ! Value must be between {minimum:g} and {maximum:g}.")
            continue
        return value


def read_student() -> dict:
    print()
    print("Enter the student's details (press Enter to accept the default).")
    print(THIN)
    name = input("  Student name              : ").strip() or "Student"
    attendance = ask_number("Attendance (%)", 0, 100, 80)
    assignment = ask_number("Assignment marks (%)", 0, 100, 70)
    internal = ask_number("Internal marks (%)", 0, 100, 65)
    study = ask_number("Study hours per day", 0, 10, 3)
    return {
        "name": name,
        "attendance": attendance,
        "assignment": assignment,
        "internal": internal,
        "study": study,
    }


def show_result(student: dict) -> None:
    bundle = P.load_bundle()
    result = P.predict_performance(
        bundle,
        student["attendance"],
        student["assignment"],
        student["internal"],
        student["study"],
    )
    label = result["label"]
    style = P.CLASS_STYLE[label]

    print()
    print(THIN)
    print("  RESULT")
    print(THIN)
    print(f"  Student      : {student['name']}")
    print(f"  Prediction   : {style['emoji']} {label.upper()}")
    print(f"  Confidence   : {result['confidence']:.1%}")
    print(f"  Readiness    : {result['score']:.1f} / 100  (weighted average of the inputs)")
    print(f"  Meaning      : {MEANING[label]}")

    print()
    print("  Probability across the four bands")
    for cls in P.CLASS_ORDER:
        pct = result["probabilities"][cls] * 100
        print(f"    {cls:<10} {bar(pct)} {pct:5.1f}%")

    print()
    print(THIN)
    print("  IMPROVEMENT PLAN")
    print(THIN)

    suggestions = P.generate_suggestions(result["inputs"], label)
    for index, item in enumerate(suggestions, start=1):
        print(f"  {index}. [{item['priority'].upper()}] {item['area']}")
        print(f"     {item['message']}")
        print(f"     Do this: {item['action']}")
        print()

    gaps = P.biggest_gaps(result["inputs"], limit=2)
    if gaps:
        print(f"  Focus first on: {', '.join(gaps)}")
    print()
    print("  Note: advisory output from a student project, not an official grade.")
    print(RULE)


MEANING = {
    "Excellent": "On track for a strong final grade.",
    "Good": "Above average - one band away from Excellent.",
    "Average": "Middle band - targeted fixes move this up quickly.",
    "Poor": "At risk - act within the next two weeks.",
}


def main() -> None:
    setup_console()
    print()
    print(RULE)
    print("  INTELLIGENT STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("  Decision Tree Classifier - 4 academic indicators")
    print(RULE)

    try:
        if "--demo" in sys.argv:
            student = {"name": "Demo Student", **DEMO_SAMPLE}
            print("\n  Running the built-in at-risk demo sample.")
        else:
            student = read_student()
        show_result(student)
    except (KeyboardInterrupt, EOFError):
        print("\n\nCancelled.")
    except FileNotFoundError as error:
        print(f"\n{error}\n")


if __name__ == "__main__":
    main()
