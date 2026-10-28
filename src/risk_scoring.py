
import pandas as pd


def calculate_risk(coordinated_pairs):
    if coordinated_pairs.empty:
        return pd.DataFrame()

    data = coordinated_pairs.copy()

    # Text similarity: 70% of the score
    similarity_score = data["similarity"] * 70

    # Timing coordination: 30% of the score
    timing_score = (
        1 - data["time_difference_seconds"] / 300
    ).clip(0, 1) * 30

    data["risk_score"] = (
        similarity_score + timing_score
    ).round(1)

    data["risk_level"] = pd.cut(
        data["risk_score"],
        bins=[-1, 39, 69, 100],
        labels=["Low", "Medium", "High"]
    )

    return data.sort_values(
        "risk_score", ascending=False
    ).reset_index(drop=True)
