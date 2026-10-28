
import pandas as pd
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

OUTPUT = Path("data/sample_posts.csv")
OUTPUT.parent.mkdir(exist_ok=True)

posts = []
post_number = 1

start_time = datetime(2026, 10, 28, 10, 0, 0)

# Scenario 1: Similar posts published within a short interval.
# This is synthetic data for testing, not real social media activity.
coordinated_messages = [
    "The city has announced a major public transport update today.",
    "A major public transport update has been announced by the city today.",
    "Today the city announced a major update to public transport.",
    "The city announced its major public transport update today.",
]

for group in range(5):
    base_time = start_time + timedelta(minutes=group * 20)

    for i in range(4):
        posts.append({
            "post_id": f"P{post_number:04}",
            "account_id": f"A{group * 4 + i + 1:03}",
            "text": coordinated_messages[i % len(coordinated_messages)],
            "timestamp": base_time + timedelta(seconds=i * 15),
            "platform": "SyntheticDemo",
            "reply_to": "",
            "location": random.choice(
                ["Region A", "Region B", "Region C"]
            ),
            "scenario": "similar_synchronized"
        })
        post_number += 1

# Scenario 2: Ordinary discussion with varied wording.
organic_messages = [
    "I took the bus to college this morning.",
    "The weather is pleasant today.",
    "Traffic was heavy near the railway station.",
    "I need to finish my assignment before Friday.",
    "The local library has extended its opening hours.",
    "Does anyone know when the next train arrives?",
    "I enjoyed the technology exhibition yesterday.",
    "Public transport can make commuting more convenient.",
    "I am looking for a good place to study.",
    "The college event starts at ten tomorrow.",
]

for i in range(30):
    posts.append({
        "post_id": f"P{post_number:04}",
        "account_id": f"A{100 + i:03}",
        "text": random.choice(organic_messages),
        "timestamp": start_time + timedelta(
            minutes=random.randint(0, 180)
        ),
        "platform": "SyntheticDemo",
        "reply_to": "",
        "location": random.choice(
            ["Region A", "Region B", "Region C", "Unknown"]
        ),
        "scenario": "organic_or_unrelated"
    })
    post_number += 1

df = pd.DataFrame(posts)

df = df.sort_values("timestamp").reset_index(drop=True)
df.to_csv(OUTPUT, index=False)

print(f"Dataset created: {OUTPUT.resolve()}")
print(f"Total posts: {len(df)}")
print(f"Unique accounts: {df['account_id'].nunique()}")
print("\nScenario counts:")
print(df["scenario"].value_counts())
print("\nFirst five rows:")
print(df.head().to_string(index=False))
