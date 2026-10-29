
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@lru_cache(maxsize=1)
def load_semantic_model():
    """Load the semantic model once and reuse it."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def rank_articles_semantically(claim, articles, min_similarity=0.25):
    """
    Rank articles by meaning rather than exact word overlap.
    Similarity measures relevance, not factual truth.
    """

    if not isinstance(claim, str) or not claim.strip() or not articles:
        return []

    valid_articles = [
        article for article in articles
        if isinstance(article, dict)
        and (article.get("title") or article.get("summary"))
    ]

    if not valid_articles:
        return []

    try:
        model = load_semantic_model()

        claim_embedding = model.encode(
            [claim.strip()],
            normalize_embeddings=True
        )

        article_texts = [
            f"{article.get('title', '')}. {article.get('summary', '')}"
            for article in valid_articles
        ]

        article_embeddings = model.encode(
            article_texts,
            normalize_embeddings=True
        )

        scores = cosine_similarity(
            claim_embedding,
            article_embeddings
        )[0]

    except Exception as error:
        print(f"Semantic matching failed: {error}")
        return []

    ranked = []

    for article, score in zip(valid_articles, scores):
        if score >= min_similarity:
            result = article.copy()
            result["semantic_score"] = round(float(score), 3)
            ranked.append(result)

    ranked.sort(
        key=lambda article: article["semantic_score"],
        reverse=True
    )

    return ranked

def rank_articles(claim, articles, min_similarity=0.02):
    """Rank articles by word overlap with the claim."""

    if not isinstance(claim, str) or not claim.strip() or not articles:
        return []

    valid_articles = [
        article for article in articles
        if isinstance(article, dict)
    ]

    if not valid_articles:
        return []

    documents = [claim.strip()]

    for article in valid_articles:
        title = str(article.get("title", ""))
        summary = str(article.get("summary", ""))
        documents.append(f"{title} {title} {summary}")

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(documents)
        scores = cosine_similarity(
            matrix[0:1],
            matrix[1:]
        ).flatten()

    except ValueError:
        return []

    ranked = []

    for article, score in zip(valid_articles, scores):
        if score >= min_similarity:
            result = article.copy()
            result["similarity_score"] = round(float(score), 3)
            ranked.append(result)

    ranked.sort(
        key=lambda article: article["similarity_score"],
        reverse=True
    )

    return ranked


def rank_articles_combined(claim, articles):
    """Combine TF-IDF and semantic relevance rankings."""

    tfidf_results = rank_articles(
        claim, articles, min_similarity=0.0
    )

    semantic_results = rank_articles_semantically(
        claim, articles, min_similarity=0.0
    )

    combined = {}

    for article in articles or []:
        if not isinstance(article, dict):
            continue

        url = article.get("url", "")
        if url:
            combined[url] = article.copy()

    for article in tfidf_results:
        url = article.get("url", "")
        if url:
            combined.setdefault(url, article.copy())
            combined[url]["similarity_score"] = article.get(
                "similarity_score", 0.0
            )

    for article in semantic_results:
        url = article.get("url", "")
        if url:
            combined.setdefault(url, article.copy())
            combined[url]["semantic_score"] = article.get(
                "semantic_score", 0.0
            )

    ranked = []

    for article in combined.values():
        tfidf_score = article.get("similarity_score", 0.0)
        semantic_score = article.get("semantic_score", 0.0)

        article["combined_score"] = round(
            0.4 * tfidf_score + 0.6 * semantic_score,
            3
        )

        ranked.append(article)

    ranked.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )

    return ranked