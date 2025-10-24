import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


MODEL_NAME_MAP = {
    "model_1": "GPT-4o",
    "model_2": "DeepSeek-R1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine judgment outputs with source metadata into a review CSV."
    )
    parser.add_argument("--csv-path", required=True, help="Path to the original harmonized CSV.")
    parser.add_argument(
        "--judgments-dir", default="judgments", help="Directory containing judgment JSON files."
    )
    parser.add_argument(
        "--output-path", default="compiled_results.csv", help="Where to write the combined CSV."
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "post_id" not in df.columns:
        raise ValueError("CSV missing required column 'post_id'.")
    # Keep only the first occurrence per post_id to avoid duplicate joins later.
    metadata = df.drop_duplicates(subset=["post_id"])
    logging.info("Loaded metadata for %d unique posts.", len(metadata))
    return metadata


def resolve_model_for_summary(
    summary_label: str,
    judgment: Dict,
) -> Optional[str]:
    mapping = judgment.get("randomized_order_mapping")
    if mapping:
        model_id = mapping.get(summary_label)
    else:
        randomized_order = judgment.get("randomized_order", [])
        index = 0 if summary_label == "summary_a" else 1
        model_id = randomized_order[index] if index < len(randomized_order) else None
    if model_id is None:
        return None
    return MODEL_NAME_MAP.get(model_id, model_id)


def collect_rows(
    metadata_df: pd.DataFrame,
    judgments_dir: Path,
) -> List[Dict]:
    rows: List[Dict] = []
    metadata_lookup = metadata_df.set_index("post_id")

    for judgment_file in sorted(judgments_dir.glob("*__gpt4o_judgment.json")):
        post_id = judgment_file.name.split("__", 1)[0]
        try:
            judgment = json.loads(judgment_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logging.warning("Skipping %s due to JSON error: %s", judgment_file, exc)
            continue

        metadata = metadata_lookup.loc[post_id] if post_id in metadata_lookup.index else None
        group = metadata["group"] if metadata is not None and "group" in metadata else None
        keyword = metadata["keyword"] if metadata is not None and "keyword" in metadata else None

        winner_label = judgment.get("winner", "")
        scores = judgment.get("scores", {})
        rationale = judgment.get("rationale", "")

        for idx, summary_label in enumerate(["summary_a", "summary_b"]):
            model_name = resolve_model_for_summary(summary_label, judgment)
            if not model_name:
                logging.warning("Could not determine model for %s (%s). Skipping.", summary_label, judgment_file)
                continue

            row = {
                "post_id": post_id,
                "group": group,
                "keyword": keyword,
                "model": model_name,
                "winner": winner_label,
                "is_winner": winner_label.lower().endswith(summary_label[-1]),
                "rationale": rationale,
                "source_file": judgment_file.name,
            }

            for score_name, values in scores.items():
                if isinstance(values, list) and idx < len(values):
                    row[f"{score_name}_score"] = values[idx]

            rows.append(row)

    logging.info("Collected %d rows from judgments.", len(rows))
    return rows


def main() -> None:
    args = parse_args()
    setup_logging()

    csv_path = Path(args.csv_path)
    judgments_dir = Path(args.judgments_dir)
    if not judgments_dir.exists():
        raise FileNotFoundError(f"Judgments directory {judgments_dir} does not exist.")

    metadata_df = load_metadata(csv_path)
    rows = collect_rows(metadata_df, judgments_dir)

    if not rows:
        logging.warning("No rows collected; no CSV will be written.")
        return

    output_df = pd.DataFrame(rows)
    output_df.to_csv(args.output_path, index=False)
    logging.info("Wrote combined results to %s", args.output_path)


if __name__ == "__main__":
    main()
