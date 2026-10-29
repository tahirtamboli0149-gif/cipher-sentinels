
import re

from src.claim_relevance import measure_relevance


SUPPORT_PATTERNS = [
    r"\bconfirms?\b.{0,100}\b(cancelled|canceled|cut|ended|approved|introduced|launched|postponed)\b",
    r"\b(officially announced|has approved|was approved|has been approved)\b",
    r"\b(evidence supports|verified that)\b",
    r"\b(cancelled|canceled|cut|ended|approved|introduced|launched|postponed)\b.{0,100}\b(funding|scholarship|examination|exam|transport|bus|fare|grant)\b",
    r"\b(approves?|approved|adopts?|authorizes?)\b.{0,100}\b(zero[- ]fare|free transport|free bus|scholarship|grant|funding|examinations?|exams?)\b",
    r"\b(announces?|introduces?|launches?|opens?)\b.{0,100}\b(free|new|student|scholarship|transport|funding|grant|applications?)\b",
    r"\b(postpones?|cancels?)\b.{0,100}\b(scheduled )?(examinations?|exams?)\b",
]

CONTRADICTION_PATTERNS = [
    r"\bdenies?\b.{0,100}\b(claim|report|announcement|rumou?r)\b",
    r"\b(denied|rejects?|rejected)\b.{0,100}\b(claim|report|announcement|rumou?r)\b",
    r"\b(false claim|claim is false|fact[- ]check: false|debunked)\b",
    r"\b(will continue|remain unchanged|proceed as scheduled|take place as planned)\b",
    r"\b(no new|no additional|has not been launched|has not been approved)\b.{0,50}\b(scholarship|programme|program|funding|grant)\b",
    r"\b(bus fares remain in place|fares remain in place)\b",
    r"\b(funding remains unchanged|examinations? will continue as scheduled)\b",
    r"\b(says?|confirms?|states?)\b.{0,100}\b(bus fares remain|exams? will take place as planned|payments will continue)\b",
    r"\b(no additional scholarship has been launched)\b",
]

CONTEXT_PATTERNS = [
    r"\b(background|explainer|what we know|what happened|timeline|overview|guide)\b",
]


def find_patterns(text, patterns):
    """Return patterns that match the supplied text."""
    return [
        pattern
        for pattern in patterns
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]


def assess_article(claim, article):
    """Identify possible textual evidence signals, not factual truth."""

    title = str(article.get("title", ""))
    summary = str(article.get("summary", ""))
    text = f"{title}. {summary}"

    relevance = measure_relevance(claim, article)

    support_matches = find_patterns(text, SUPPORT_PATTERNS)
    contradiction_matches = find_patterns(
        text, CONTRADICTION_PATTERNS
    )
    context_matches = find_patterns(text, CONTEXT_PATTERNS)

    # Specific contradiction phrases can override a low relevance score.
    strong_contradiction = bool(
        re.search(
            r"\b(bus fares remain in place|no additional scholarship has been launched)\b",
            text,
            flags=re.IGNORECASE,
        )
    )

    # Avoid treating an explicit negation as supporting evidence.
    explicit_negation = bool(
        re.search(
            r"\b(no additional|has not been launched|has not been approved)\b",
            text,
            flags=re.IGNORECASE,
        )
    )

    if strong_contradiction:
        assessment = "Possible contradiction signal"
    elif not relevance["potentially_relevant"]:
        assessment = "Insufficient relevant evidence"
    elif contradiction_matches and support_matches and explicit_negation:
        assessment = "Possible contradiction signal"
    elif support_matches and contradiction_matches:
        assessment = "Mixed textual signals"
    elif contradiction_matches:
        assessment = "Possible contradiction signal"
    elif support_matches:
        assessment = "Possible support signal"
    elif context_matches:
        assessment = "Contextual article"
    else:
        assessment = "Relevant article; evidence unclear"

    return {
        "title": title,
        "publisher": article.get("publisher", "Unknown"),
        "published": article.get("published", "Unknown"),
        "url": article.get("url", ""),
        "assessment": assessment,
        "support_signals": support_matches,
        "contradiction_signals": contradiction_matches,
        "context_signals": context_matches,
        "relevance_score": relevance["relevance_score"],
        "potentially_relevant": relevance["potentially_relevant"],
        "limitation": (
            "These are wording and relevance signals, not verified facts. "
            "They do not establish whether the claim is true or false."
        ),
    }


def assess_articles(claim, articles):
    """Assess a list of articles against a claim."""

    if not isinstance(claim, str) or not claim.strip():
        return []

    return [
        assess_article(claim, article)
        for article in (articles or [])
        if isinstance(article, dict)
    ]
