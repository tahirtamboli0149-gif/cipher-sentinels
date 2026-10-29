
import time
import requests
import feedparser
from urllib.parse import quote


def search_news(query, max_results=10):
    """
    Search Google News RSS for articles related to a claim.
    Returns article titles, publishers, dates, and URLs.
    """

    if not isinstance(query, str) or not query.strip():
        return []

    query = query.strip()
    max_results = max(1, min(int(max_results), 25))

    url = (
        "https://news.google.com/rss/search"
        f"?q={quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    )

    headers = {
        "User-Agent": "CipherSentinels/1.0"
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )
        response.raise_for_status()

        feed = feedparser.parse(response.content)
        articles = []

        for entry in feed.entries[:max_results]:
            source = entry.get("source", {})

            articles.append({
                "title": entry.get("title", "No title"),
                "publisher": source.get("title", "Unknown"),
                "published": entry.get("published", "Unknown"),
                "url": entry.get("link", ""),
                "summary": entry.get("summary", ""),
            })

        # Remove duplicate links.
        unique_articles = []
        seen_urls = set()

        for article in articles:
            article_url = article["url"]

            if article_url and article_url not in seen_urls:
                seen_urls.add(article_url)
                unique_articles.append(article)

        return unique_articles

    except requests.RequestException as error:
        print(f"News search request failed: {error}")
        return []

    except Exception as error:
        print(f"Could not process news results: {error}")
        return []
