
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_posts(df, threshold=0.35):
    """
    Find pairs of posts with similar wording.

    Returns a DataFrame containing:
    post IDs, accounts, similarity score, and publication times.
    """

    required_columns = {
        "post_id", "account_id", "text", "timestamp"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )

    if len(df) < 2:
        return pd.DataFrame()

    data = df.copy()

    data["text"] = data["text"].fillna("").astype(str).str.strip()
    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )

    data = data[data["text"] != ""].reset_index(drop=True)

    if len(data) < 2:
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    try:
        vectors = vectorizer.fit_transform(data["text"])
    except ValueError:
        # For example, all posts may be empty or contain only stop words.
        return pd.DataFrame()

    similarity_matrix = cosine_similarity(vectors)

    results = []

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            score = float(similarity_matrix[i, j])

            if score >= threshold:
                results.append({
                    "post_id_1": data.loc[i, "post_id"],
                    "account_id_1": data.loc[i, "account_id"],
                    "text_1": data.loc[i, "text"],
                    "timestamp_1": data.loc[i, "timestamp"],

                    "post_id_2": data.loc[j, "post_id"],
                    "account_id_2": data.loc[j, "account_id"],
                    "text_2": data.loc[j, "text"],
                    "timestamp_2": data.loc[j, "timestamp"],

                    "similarity": round(score, 3)
                })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values(
            "similarity", ascending=False
        ).reset_index(drop=True)

    return results_df
