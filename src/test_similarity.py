
import pandas as pd
from similarity import find_similar_posts

df = pd.read_csv("data/sample_posts.csv")

results = find_similar_posts(df, threshold=0.35)

print("\nCIPHER SENTINELS — TEXT SIMILARITY TEST")
print("=" * 50)

print(f"Total posts analyzed: {len(df)}")
print(f"Similar post pairs found: {len(results)}")

if not results.empty:
    print("\nTop 10 similar pairs:")

    print(
        results[
            [
                "account_id_1",
                "account_id_2",
                "similarity",
                "text_1",
                "text_2"
            ]
        ].head(10).to_string(index=False)
    )
else:
    print("No similar pairs found at this threshold.")
