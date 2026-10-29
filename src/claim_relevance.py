
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@lru_cache(maxsize=1)
def load_model():
    """Load the existing semantic embedding model once."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def measure_relevance(claim, article):
    """
    Estimate semantic relevance between a claim and an article.

    This measures similarity, not truth or contradiction.
    """

    if not isinstance(claim, str) or not claim.strip():
        return {
            "relevance_score": 0.0,
            "potentially_relevant": False,
        }

    title = str(article.get("title", ""))
    summary = str(article.get("summary", ""))
    article_text = f"{title}. {summary}".strip()

    if not article_text or article_text == ".":
        return {
            "relevance_score": 0.0,
            "potentially_relevant": False,
        }

    model = load_model()

    embeddings = model.encode(
        [claim.strip(), article_text],
        normalize_embeddings=True,
    )

    score = float(
        cosine_similarity(
            [embeddings[0]],
            [embeddings[1]],
        )[0][0]
    )

    return {
        "relevance_score": round(score, 3),
        "potentially_relevant": score >= 0.35,
    }
