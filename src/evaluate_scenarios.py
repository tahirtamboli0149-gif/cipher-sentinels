
import pandas as pd
from itertools import combinations

from similarity import find_similar_posts
from coordination import find_coordination
from risk_scoring import calculate_risk


# 1. Load the dataset
posts = pd.read_csv("data/sample_posts.csv")

required_columns = {
    "post_id", "account_id", "text", "timestamp", "scenario"
}

missing = required_columns - set(posts.columns)

if missing:
    raise ValueError(
        f"Dataset is missing columns: {sorted(missing)}"
    )

posts["timestamp"] = pd.to_datetime(
    posts["timestamp"], errors="coerce"
)

# Normalize IDs so comparisons use the same type
posts["post_id"] = posts["post_id"].astype(str)

# 2. Run the detection pipeline
similar = find_similar_posts(
    posts, threshold=0.35
)

coordinated = find_coordination(
    similar, time_window_minutes=5
)

results = calculate_risk(coordinated)

# 3. Reconstruct synchronized groups.
# A gap greater than 5 minutes starts a new group.
sync_posts = (
    posts[posts["scenario"] == "similar_synchronized"]
    .dropna(subset=["timestamp"])
    .sort_values("timestamp")
)

groups = []
current_group = []
previous_timestamp = None

for _, post in sync_posts.iterrows():
    timestamp = post["timestamp"]

    if previous_timestamp is not None:
        gap = (timestamp - previous_timestamp).total_seconds()

        if gap > 5 * 60:
            groups.append(current_group)
            current_group = []

    current_group.append(str(post["post_id"]))
    previous_timestamp = timestamp

if current_group:
    groups.append(current_group)

# 4. Generate expected positive pairs within each group
expected = set()

for group in groups:
    for post_a, post_b in combinations(group, 2):
        expected.add(tuple(sorted((post_a, post_b))))

# 5. Compare detected pairs with expected pairs
detected = set()
organic_flags = 0
other_flags = 0

scenario_by_post = posts.set_index("post_id")["scenario"].to_dict()

if not results.empty:
    for _, row in results.iterrows():
        post_a = str(row["post_id_1"])
        post_b = str(row["post_id_2"])
        pair = tuple(sorted((post_a, post_b)))

        scenario_a = scenario_by_post.get(post_a)
        scenario_b = scenario_by_post.get(post_b)

        if pair in expected:
            detected.add(pair)
        elif (
            scenario_a == "organic_or_unrelated"
            and scenario_b == "organic_or_unrelated"
        ):
            organic_flags += 1
        else:
            other_flags += 1

# 6. Calculate evaluation metrics
true_positive = len(detected)
false_negative = len(expected - detected)
false_positive = organic_flags + other_flags

precision = (
    true_positive / (true_positive + false_positive)
    if true_positive + false_positive else 0
)

recall = (
    true_positive / (true_positive + false_negative)
    if true_positive + false_negative else 0
)

print("\nCIPHER SENTINELS — GROUP-AWARE EVALUATION")
print("=" * 50)

print(f"Posts analyzed: {len(posts)}")
print(f"Reconstructed synchronized groups: {len(groups)}")
print(f"Expected within-group pairs: {len(expected)}")
print(f"Candidate pairs flagged: {len(results)}")
print(f"Correctly detected expected pairs: {true_positive}")
print(f"Missed expected pairs: {false_negative}")
print(f"Other flagged pairs: {false_positive}")

print(f"\nPrecision: {precision:.1%}")
print(f"Recall: {recall:.1%}")

print(f"\nOrganic-organic flagged pairs: {organic_flags}")
print(f"Other flagged pairs: {other_flags}")

print(
    "\nNOTE: This is a preliminary evaluation on synthetic data. "
    "Groups are reconstructed using timestamp gaps, so these metrics "
    "are not evidence of real-world misinformation detection accuracy."
)
