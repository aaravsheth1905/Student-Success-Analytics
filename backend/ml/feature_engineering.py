import numpy as np

def build_features(
    hours_conducted,
    hours_attended,
    total_planned_hours,
    weekly_hours,
    required_percentage,
    semester_weeks,
    K
):

    if hours_conducted == 0:
        raise ValueError("Hours conducted cannot be zero")

    current_percentage = (hours_attended / hours_conducted) * 100
    hours_missed = hours_conducted - hours_attended

    minimum_required_hours = (required_percentage / 100) * total_planned_hours
    maximum_allowed_miss = total_planned_hours - minimum_required_hours
    remaining_allowed_miss = maximum_allowed_miss - hours_missed

    miss_ratio = (
        hours_missed / maximum_allowed_miss
        if maximum_allowed_miss > 0 else 1
    )

    buffer_ratio = remaining_allowed_miss / total_planned_hours
    attendance_gap = required_percentage - current_percentage
    remaining_weeks = semester_weeks - (hours_conducted // weekly_hours)

    feature_vector = np.array([[
        current_percentage,
        miss_ratio,
        buffer_ratio,
        attendance_gap,
        remaining_weeks,
        weekly_hours,
        required_percentage,
        K
    ]])

    return feature_vector