
import pandas as pd


def find_coordination(similar_pairs, time_window_minutes=5):
    if similar_pairs.empty:
        return pd.DataFrame()

    data = similar_pairs.copy()

    data["timestamp_1"] = pd.to_datetime(
        data["timestamp_1"], errors="coerce"
    )
    data["timestamp_2"] = pd.to_datetime(
        data["timestamp_2"], errors="coerce"
    )

    data = data.dropna(
        subset=["timestamp_1", "timestamp_2"]
    )

    data["time_difference_seconds"] = (
        data["timestamp_1"] - data["timestamp_2"]
    ).abs().dt.total_seconds()

    data = data[
        data["time_difference_seconds"]
        <= time_window_minutes * 60
    ].copy()

    if data.empty:
        return pd.DataFrame()

    data["coordination_score"] = (
        data["similarity"]
        * (
            1 - data["time_difference_seconds"]
            / (time_window_minutes * 60)
        )
    ).round(3)

    data = data.sort_values(
        "coordination_score", ascending=False
    ).reset_index(drop=True)

    return data
