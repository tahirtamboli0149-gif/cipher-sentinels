
import pandas as pd

from similarity import find_similar_posts
from coordination import find_coordination
from risk_scoring import calculate_risk


posts = pd.read_csv("data/sample_posts.csv")

similar_pairs = find_similar_posts(
    posts, threshold=0.35
)

coordinated_pairs = find_coordination(
    similar_pairs, time_window_minutes=5
)

results = calculate_risk(coordinated_pairs)

print("\nCIPHER SENTINELS — RISK SCORING TEST")
print("=" * 50)
print(f"Posts analyzed: {len(posts)}")
print(f"Candidate pairs: {len(results)}")

if not results.empty:
    print("\nRisk level counts:")
    print(results["risk_level"].value_counts())

    print("\nTop 10 pairs:")
    columns = [
        "account_id_1",
        "account_id_2",
        "similarity",
        "time_difference_seconds",
        "risk_score",
        "risk_level",
    ]

    print(results[columns].head(10).to_string(index=False))
