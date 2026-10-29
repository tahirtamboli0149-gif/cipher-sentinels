
import pandas as pd
import networkx as nx


def detect_campaigns(
    coordinated_pairs,
    min_accounts=2,
    min_connections=1
):
    """
    Group connected accounts into candidate campaigns.

    Accounts are connected when their posts have already
    passed the similarity and coordination checks.
    """

    if coordinated_pairs.empty:
        return pd.DataFrame(), pd.DataFrame()

    required_columns = {
        "account_id_1",
        "account_id_2",
        "post_id_1",
        "post_id_2",
        "similarity",
        "coordination_score"
    }

    missing = required_columns - set(coordinated_pairs.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )

    graph = nx.Graph()

    for _, row in coordinated_pairs.iterrows():
        account_1 = str(row["account_id_1"])
        account_2 = str(row["account_id_2"])

        if account_1 == account_2:
            continue

        graph.add_edge(
            account_1,
            account_2,
            similarity=float(row["similarity"]),
            coordination_score=float(row["coordination_score"]),
            post_id_1=row["post_id_1"],
            post_id_2=row["post_id_2"]
        )

    campaign_summaries = []
    account_memberships = []

    campaign_number = 0

    for accounts in nx.connected_components(graph):
        subgraph = graph.subgraph(accounts)

        if len(accounts) < min_accounts:
            continue

        if subgraph.number_of_edges() < min_connections:
            continue

        campaign_number += 1
        campaign_id = f"CAMP-{campaign_number:03d}"

        all_posts = set()
        similarities = []
        coordination_scores = []

        for account_1, account_2, details in subgraph.edges(data=True):
            all_posts.add(details["post_id_1"])
            all_posts.add(details["post_id_2"])

            similarities.append(details["similarity"])
            coordination_scores.append(
                details["coordination_score"]
            )

        average_similarity = sum(similarities) / len(similarities)
        average_coordination = (
            sum(coordination_scores) / len(coordination_scores)
        )

        campaign_summaries.append({
            "campaign_id": campaign_id,
            "account_count": len(accounts),
            "connection_count": subgraph.number_of_edges(),
            "related_post_count": len(all_posts),
            "average_similarity": round(average_similarity, 3),
            "average_coordination_score": round(
                average_coordination, 3
            ),
            "accounts": ", ".join(sorted(accounts)),
            "risk_level": (
                "High" if average_coordination >= 0.5
                else "Medium" if average_coordination >= 0.2
                else "Low"
            )
        })

        for account in sorted(accounts):
            account_memberships.append({
                "campaign_id": campaign_id,
                "account_id": account,
                "connection_count": subgraph.degree(account)
            })

    campaigns_df = pd.DataFrame(campaign_summaries)
    memberships_df = pd.DataFrame(account_memberships)

    if not campaigns_df.empty:
        campaigns_df = campaigns_df.sort_values(
            ["account_count", "average_coordination_score"],
            ascending=False
        ).reset_index(drop=True)

    return campaigns_df, memberships_df
