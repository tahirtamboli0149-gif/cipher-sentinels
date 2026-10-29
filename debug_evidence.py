
import pandas as pd
from src.evidence_assessment import assess_article

df = pd.read_csv("data/evidence_unseen_test.csv")

for _, row in df.iterrows():
    article = {
        "title": row["article_title"],
        "summary": row["article_summary"],
        "publisher": "Test",
    }

    result = assess_article(row["claim"], article)

    print(f"\nExpected:   {row['label']}")
    print(f"Predicted:  {result['assessment']}")
    print(f"Relevance:  {result['relevance_score']:.3f}")
    print(f"Title:      {row['article_title']}")
