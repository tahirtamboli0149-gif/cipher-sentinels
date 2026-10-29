
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make the src folder available for imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from similarity import find_similar_posts
from coordination import find_coordination
from risk_scoring import calculate_risk
from campaign_detection import detect_campaigns

from news_search import search_news
from evidence_matching import rank_articles_combined
from evidence_assessment import assess_articles


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CIPHER SENTINELS",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ CIPHER SENTINELS")
st.subheader("AI-Powered Coordinated Disinformation Detection")

st.caption(
    "Identify patterns of similar content and coordinated timing. "
    "Flags indicate potential coordination, not proof of misinformation."
)


# --------------------------------------------------
# SOCIAL MEDIA CLAIM VERIFIER
# --------------------------------------------------

st.divider()
st.header("🔎 Verify a Social Media Post")

st.write(
    "Paste a social media post or claim to search live news sources "
    "and examine potentially supporting or contradicting evidence."
)

claim_text = st.text_area(
    "Enter the post or claim",
    height=130,
    placeholder=(
        "Example: The university has postponed all upcoming examinations."
    ),
    key="claim_input",
)

if st.button("🔍 Search Live Evidence", type="primary"):

    if not claim_text.strip():
        st.warning("Please enter a post or claim first.")

    else:
        try:
            with st.spinner("Searching live news sources..."):
                articles = search_news(
                    claim_text.strip(),
                    max_results=10,
                )

            if not articles:
                st.warning(
                    "No news articles were retrieved. Try a shorter claim "
                    "or search again later."
                )

            else:
                with st.spinner("Analyzing the retrieved articles..."):
                    ranked_articles = rank_articles_combined(
                        claim_text.strip(),
                        articles,
                    )

                    assessments = assess_articles(
                        claim_text.strip(),
                        articles,
                    )

                assessment_by_url = {
                    item.get("url", ""): item
                    for item in assessments
                }

                st.subheader("Evidence Overview")

                st.write(
                    f"**Articles retrieved:** {len(articles)}"
                )

                st.caption(
                    "Relevance scores measure how closely an article "
                    "matches the claim. They are not truth probabilities."
                )

                if not ranked_articles:
                    st.info(
                        "Articles were retrieved, but none could be ranked."
                    )

                else:
                    for article in ranked_articles:

                        article_url = article.get("url", "")

                        evidence = assessment_by_url.get(
                            article_url,
                            {},
                        )

                        article_title = article.get(
                            "title",
                            "Untitled article",
                        )

                        with st.expander(str(article_title)):

                            st.write(
                                "**Publisher:**",
                                article.get("publisher", "Unknown"),
                            )

                            st.write(
                                "**Published:**",
                                article.get("published", "Unknown"),
                            )

                            st.write(
                                "**Preliminary evidence signal:**",
                                evidence.get(
                                    "assessment",
                                    "Relevant article; evidence unclear",
                                ),
                            )

                            relevance_score = article.get(
                                "combined_score",
                                article.get("similarity_score", 0),
                            )

                            try:
                                relevance_score = float(relevance_score)
                            except (TypeError, ValueError):
                                relevance_score = 0.0

                            st.write(
                                "**Relevance score:**",
                                round(relevance_score, 3),
                            )

                            summary = article.get("summary", "")

                            if summary:
                                st.write("**Available summary:**")
                                st.write(summary)

                            if article_url:
                                st.markdown(
                                    f"[Read the original article]"
                                    f"({article_url})"
                                )

                            st.caption(
                                evidence.get(
                                    "limitation",
                                    "Automated signals are not proof. "
                                    "Review the original source before "
                                    "drawing a conclusion.",
                                )
                            )

                st.warning(
                    "This is a preliminary evidence search, not a "
                    "definitive fact-check. News headlines and summaries "
                    "may be incomplete or misleading. A missing result "
                    "does not mean the claim is false."
                )

        except Exception as error:
            st.error(
                f"Evidence search failed: {error}"
            )


# --------------------------------------------------
# CAMPAIGN DETECTION SETTINGS
# --------------------------------------------------

st.divider()

with st.sidebar:

    st.header("Detection settings")

    similarity_threshold = st.slider(
        "Text similarity threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
    )

    time_window = st.slider(
        "Time window (minutes)",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
    )

    st.divider()

    st.write("**Expected CSV columns**")

    st.code(
        "post_id, account_id, text, timestamp"
    )


# --------------------------------------------------
# LOAD POSTS DATASET
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a posts CSV file",
    type=["csv"],
)

if uploaded_file is not None:

    try:
        posts = pd.read_csv(uploaded_file)

    except Exception as error:
        st.error(f"Could not read CSV: {error}")
        st.stop()

else:

    st.info(
        "Using the built-in sample dataset. "
        "Upload your own CSV to analyze it."
    )

    sample_path = ROOT / "data" / "sample_posts.csv"

    if not sample_path.exists():
        st.error(
            "Sample dataset not found. Check data/sample_posts.csv."
        )
        st.stop()

    posts = pd.read_csv(sample_path)


required_columns = {
    "post_id",
    "account_id",
    "text",
    "timestamp",
}

missing = required_columns - set(posts.columns)

if missing:
    st.error(
        "Missing required columns: "
        + ", ".join(sorted(missing))
    )
    st.stop()


# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

st.markdown("### Dataset preview")

st.dataframe(
    posts.head(10),
    use_container_width=True,
)


# --------------------------------------------------
# ANALYZE SIMILARITY, COORDINATION AND RISK
# --------------------------------------------------

with st.spinner(
    "Analyzing text similarity and timing patterns..."
):

    similar_pairs = find_similar_posts(
        posts,
        threshold=similarity_threshold,
    )

    coordinated_pairs = find_coordination(
        similar_pairs,
        time_window_minutes=time_window,
    )

    results = calculate_risk(
        coordinated_pairs,
    )

campaigns, memberships = detect_campaigns(
    coordinated_pairs
)


# --------------------------------------------------
# RISK COUNTS
# --------------------------------------------------

high_count = 0
medium_count = 0
low_count = 0

if not results.empty:

    levels = results["risk_level"].astype(str)

    high_count = int(
        (levels == "High").sum()
    )

    medium_count = int(
        (levels == "Medium").sum()
    )

    low_count = int(
        (levels == "Low").sum()
    )


# --------------------------------------------------
# ANALYSIS SUMMARY
# --------------------------------------------------

st.markdown("### Analysis summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Posts analyzed",
    len(posts),
)

c2.metric(
    "Similar pairs",
    len(similar_pairs),
)

c3.metric(
    "Time-matched pairs",
    len(results),
)

c4.metric(
    "High-risk pairs",
    high_count,
)


# --------------------------------------------------
# RISK LEVEL CHART
# --------------------------------------------------

st.markdown("### Candidate pair risk levels")

risk_summary = pd.DataFrame(
    {
        "Risk level": ["High", "Medium", "Low"],
        "Number of pairs": [
            high_count,
            medium_count,
            low_count,
        ],
    }
)

st.bar_chart(
    risk_summary.set_index("Risk level"),
    y="Number of pairs",
)


# --------------------------------------------------
# CAMPAIGN INTELLIGENCE
# --------------------------------------------------

st.markdown("### Campaign Intelligence")

if campaigns.empty:

    st.info(
        "No candidate campaigns detected with the current settings. "
        "Try adjusting the similarity threshold or time window."
    )

else:

    campaign_col1, campaign_col2, campaign_col3 = st.columns(3)

    campaign_col1.metric(
        "Candidate campaigns",
        len(campaigns),
    )

    campaign_col2.metric(
        "Accounts involved",
        campaigns["account_count"].sum(),
    )

    campaign_col3.metric(
        "Related connections",
        campaigns["connection_count"].sum(),
    )

    campaign_columns = [
        "campaign_id",
        "account_count",
        "connection_count",
        "related_post_count",
        "average_similarity",
        "average_coordination_score",
        "risk_level",
        "accounts",
    ]

    available_campaign_columns = [
        column
        for column in campaign_columns
        if column in campaigns.columns
    ]

    st.dataframe(
        campaigns[available_campaign_columns],
        use_container_width=True,
        hide_index=True,
    )

    if "campaign_id" in campaigns.columns:

        selected_campaign_id = st.selectbox(
            "Investigate a campaign",
            options=campaigns["campaign_id"].tolist(),
            format_func=lambda campaign_id: (
                f"{campaign_id} — "
                f"{campaigns.loc[campaigns['campaign_id'] == campaign_id, 'account_count'].iloc[0]} "
                "accounts"
            ),
        )

        selected_campaign = campaigns[
            campaigns["campaign_id"] == selected_campaign_id
        ].iloc[0]

        st.write("**Accounts in this campaign:**")

        st.write(
            selected_campaign.get("accounts", "Not available")
        )

    st.caption(
        "Campaigns are groups of accounts connected by similar, "
        "closely timed posts. Risk categories are preliminary "
        "investigation signals, not proof of disinformation."
    )


# --------------------------------------------------
# CANDIDATE COORDINATED POSTS
# --------------------------------------------------

st.markdown("### Candidate coordinated posts")

if results.empty:

    st.success(
        "No pairs matched the current similarity and timing settings."
    )

else:

    display_columns = [
        "account_id_1",
        "account_id_2",
        "similarity",
        "time_difference_seconds",
        "risk_score",
        "risk_level",
        "text_1",
        "text_2",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in results.columns
    ]

    st.dataframe(
        results[available_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download analysis results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="cipher_sentinels_results.csv",
        mime="text/csv",
    )


st.warning(
    "Interpretation: a high score means strong similarity and close timing "
    "under the selected settings. It does not establish that a claim is false, "
    "that users coordinated intentionally, or that an account is malicious. "
    "Review the original context before drawing conclusions."
)


# --------------------------------------------------
# POSTING ACTIVITY TIMELINE
# --------------------------------------------------

st.markdown("### Posting Activity Timeline")

timeline_data = posts.copy()

timeline_data["timestamp"] = pd.to_datetime(
    timeline_data["timestamp"],
    errors="coerce",
)

timeline_data = timeline_data.dropna(
    subset=["timestamp"]
)

if timeline_data.empty:

    st.info(
        "No valid timestamps are available to display."
    )

else:

    timeline_data["time_bucket"] = (
        timeline_data["timestamp"].dt.floor("5min")
    )

    activity = (
        timeline_data.groupby("time_bucket")
        .size()
        .reset_index(name="post_count")
        .sort_values("time_bucket")
    )

    fig = px.bar(
        activity,
        x="time_bucket",
        y="post_count",
        title="Posts per 5-minute interval",
        labels={
            "time_bucket": "Time",
            "post_count": "Number of posts",
        },
    )

    fig.update_layout(
        height=350,
        xaxis_title="Posting time",
        yaxis_title="Number of posts",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "Posting bursts can be a useful investigation signal, "
        "but high activity alone does not establish coordination."
    )


# --------------------------------------------------
# INVESTIGATION EVIDENCE
# --------------------------------------------------

st.markdown("### Investigation Evidence")

if results.empty:

    st.info(
        "No candidate pairs are available to investigate."
    )

else:

    selected_index = st.selectbox(
        "Select a candidate pair",
        options=list(range(len(results))),
        format_func=lambda i: (
            f"{results.iloc[i]['account_id_1']} ↔ "
            f"{results.iloc[i]['account_id_2']} | "
            f"Risk score: {results.iloc[i]['risk_score']}"
        ),
    )

    selected = results.iloc[selected_index]

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Post A**")

        st.write(
            f"Account: {selected['account_id_1']}"
        )

        st.write(
            selected["text_1"]
        )

        st.caption(
            f"Posted: {selected['timestamp_1']}"
        )

    with col2:

        st.markdown("**Post B**")

        st.write(
            f"Account: {selected['account_id_2']}"
        )

        st.write(
            selected["text_2"]
        )

        st.caption(
            f"Posted: {selected['timestamp_2']}"
        )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Text similarity",
        f"{selected['similarity']:.1%}",
    )

    m2.metric(
        "Time difference",
        f"{selected['time_difference_seconds']:.0f} seconds",
    )

    m3.metric(
        "Coordination score",
        f"{selected['risk_score']:.1f}/100",
    )

    st.info(
        "Why flagged: the posts have similar wording and were published "
        "within the configured time window. These signals justify further "
        "review, but do not prove misinformation or intentional coordination."
    )


# --------------------------------------------------
# ACCOUNT COORDINATION NETWORK
# --------------------------------------------------

st.markdown("### Account Coordination Network")

st.caption(
    "Connections represent pairs with similar posts published "
    "within the selected time window. They are leads for review, "
    "not proof of malicious coordination."
)

if results.empty:

    st.info(
        "No connections to display. Try adjusting the similarity "
        "threshold or time window in the sidebar."
    )

else:

    graph = nx.Graph()

    for _, row in results.iterrows():

        account_a = str(row["account_id_1"])
        account_b = str(row["account_id_2"])

        graph.add_edge(
            account_a,
            account_b,
            score=float(row["risk_score"]),
            similarity=float(row["similarity"]),
        )

    positions = nx.spring_layout(
        graph,
        seed=42,
        k=0.9,
        iterations=60,
    )

    # Build the connection lines
    edge_x = []
    edge_y = []

    for source, target in graph.edges():

        x0, y0 = positions[source]
        x1, y1 = positions[target]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="#64748b"),
        hoverinfo="none",
        name="Connections",
    )

    # Build the account nodes
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_hover = []

    for account in graph.nodes():

        x, y = positions[account]
        degree = graph.degree(account)

        node_x.append(x)
        node_y.append(y)
        node_text.append(account)
        node_sizes.append(14 + degree * 4)

        node_hover.append(
            f"Account: {account}<br>"
            f"Connected accounts: {degree}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_sizes,
            colorscale="Blues",
            showscale=False,
            line=dict(width=1, color="white"),
        ),
        name="Accounts",
    )

    figure = go.Figure(
        data=[edge_trace, node_trace]
    )

    figure.update_layout(
        height=600,
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    st.markdown(
        "**Graph summary:** "
        f"{graph.number_of_nodes()} accounts · "
        f"{graph.number_of_edges()} candidate connections"
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.caption(
    "CIPHER SENTINELS | Hackathon prototype | Synthetic sample data"
)
