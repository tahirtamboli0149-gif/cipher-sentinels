
import pandas as pd

from similarity import find_similar_posts
from coordination import find_coordination


posts = pd.read_csv("data/sample_posts.csv")

similar_pairs = find_similar_posts(
    posts, threshold=0.35
)

coordinated_pairs = find_coordination(
    similar_pairs, time_window_minutes=5
)

print("\nCIPHER SENTINELS — COORDINATION TEST")
print("=" * 50)
print(f"Posts analyzed: {len(posts)}")
print(f"Similar pairs: {len(similar_pairs)}")
print(f"Pairs posted within 5 minutes: {len(coordinated_pairs)}")

if not coordinated_pairs.empty:
    print("\nTop 10 candidate coordinated pairs:")

    columns = [
        "account_id_1",
        "account_id_2",
        "similarity",
        "time_difference_seconds",
        "coordination_score",
    ]

    print(
        coordinated_pairs[columns]
        .head(10)
        .to_string(index=False)
    )
