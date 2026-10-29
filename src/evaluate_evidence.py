
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from src.evidence_assessment import assess_article


DATASET_PATH = "data/evidence_unseen_test.csv"


def predict_label(claim, article):
    """Map heuristic signals to the dataset's three labels."""

    result = assess_article(claim, article)
    assessment = result["assessment"]

    if assessment == "Possible support signal":
        return "supports"

    if assessment == "Possible contradiction signal":
        return "contradicts"

    return "context"


def main():
    df = pd.read_csv(DATASET_PATH)

    required_columns = {
        "claim",
        "article_title",
        "article_summary",
        "label",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain: {sorted(required_columns)}"
        )

    predictions = []

    for _, row in df.iterrows():
        article = {
            "title": row["article_title"],
            "summary": row["article_summary"],
            "publisher": "Evaluation dataset",
        }

        predictions.append(
            predict_label(row["claim"], article)
        )

    df["predicted_label"] = predictions

    print("\nEVIDENCE ASSESSMENT EVALUATION")
    print("=" * 40)
    print(f"Total examples: {len(df)}")
    print(
        f"Accuracy: "
        f"{accuracy_score(df['label'], df['predicted_label']):.2%}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            df["label"],
            df["predicted_label"],
            labels=["supports", "contradicts", "context"],
            zero_division=0,
        )
    )

    print("\nIndividual predictions:")
    print(
        df[
            [
                "claim",
                "article_title",
                "label",
                "predicted_label",
            ]
        ].to_string(index=False)
    )

    output_path = "data/evidence_evaluation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

    print(
        "\nWARNING: These examples are manually constructed. "
        "This score is not evidence of real-world fact-checking accuracy."
    )


if __name__ == "__main__":
    main()
