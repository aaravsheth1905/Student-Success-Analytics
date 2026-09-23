import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os

# -------------------------
# PARAMETERS
# -------------------------

NUM_STUDENTS = 15000
MAX_WEEKS = 18
MIN_WEEKS = 12

random.seed(42)
np.random.seed(42)


# -------------------------
# SIMULATION ENGINE
# -------------------------

def simulate_student():

    semester_weeks = random.randint(MIN_WEEKS, MAX_WEEKS)
    weekly_hours = random.randint(2, 6)
    required_percentage = random.choice([75, 80])
    absentee_prob = random.uniform(0.05, 0.4)

    total_planned_hours = semester_weeks * weekly_hours

    hours_attended = 0
    hours_conducted = 0

    weekly_records = []

    for week in range(1, semester_weeks + 1):

        # small behavioural drift
        drift = random.uniform(-0.02, 0.02)
        absentee_prob = min(max(absentee_prob + drift, 0.01), 0.6)

        for _ in range(weekly_hours):
            hours_conducted += 1
            if random.random() > absentee_prob:
                hours_attended += 1

        weekly_records.append(
            (week, hours_conducted, hours_attended, absentee_prob)
        )

    final_percentage = (hours_attended / total_planned_hours) * 100
    final_failure = 1 if final_percentage < required_percentage else 0

    return {
        "semester_weeks": semester_weeks,
        "weekly_hours": weekly_hours,
        "required_percentage": required_percentage,
        "total_planned_hours": total_planned_hours,
        "weekly_records": weekly_records,
        "final_failure": final_failure
    }


# -------------------------
# FEATURE ENGINEERING
# -------------------------

def build_dataset():

    rows = []

    for _ in range(NUM_STUDENTS):

        student = simulate_student()

        semester_weeks = student["semester_weeks"]
        weekly_hours = student["weekly_hours"]
        required_percentage = student["required_percentage"]
        total_planned_hours = student["total_planned_hours"]
        final_failure = student["final_failure"]

        for week, conducted, attended, absentee_prob in student["weekly_records"]:

            current_percentage = (attended / conducted) * 100
            hours_missed = conducted - attended

            minimum_required_hours = (required_percentage / 100) * total_planned_hours
            maximum_allowed_miss = total_planned_hours - minimum_required_hours
            remaining_allowed_miss = maximum_allowed_miss - hours_missed

            miss_ratio = (
                hours_missed / maximum_allowed_miss
                if maximum_allowed_miss > 0
                else 1
            )

            buffer_ratio = remaining_allowed_miss / total_planned_hours
            attendance_gap = required_percentage - current_percentage
            remaining_weeks = max(semester_weeks - week, 0)

            rows.append([
                current_percentage,
                miss_ratio,
                buffer_ratio,
                attendance_gap,
                remaining_weeks,
                weekly_hours,
                required_percentage,
                final_failure
            ])

    columns = [
        "current_percentage",
        "miss_ratio",
        "buffer_ratio",
        "attendance_gap",
        "remaining_weeks",
        "weekly_hours",
        "required_percentage",
        "final_failure"
    ]

    return pd.DataFrame(rows, columns=columns)


# -------------------------
# TRAINING
# -------------------------

def train_models():

    df = build_dataset()

    X = df.drop(columns=["final_failure"])
    y = df["final_failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    final_model = LogisticRegression(max_iter=1000)
    final_model.fit(X_train, y_train)

    roc_auc = roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])
    print("Final Model ROC-AUC:", roc_auc)
    print("\nFinal Failure Rate:", df["final_failure"].mean())

    print("\nFinal Model Coefficients:")
    for feature, coef in zip(X.columns, final_model.coef_[0]):
        print(feature, round(coef, 4))

    os.makedirs("backend/ml/models", exist_ok=True)
    joblib.dump(final_model, "backend/ml/models/final_risk_model.pkl")
    print("Final risk model saved successfully.")


if __name__ == "__main__":
    train_models()

