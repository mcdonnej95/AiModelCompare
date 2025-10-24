import argparse
import json
import logging
import os
from pathlib import Path
from typing import Iterable

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate Chinese summaries and judgment files into English."
    )
    parser.add_argument("--summaries-dir", default="summaries", help="Directory with Chinese summaries.")
    parser.add_argument("--judgments-dir", default="judgments", help="Directory with Chinese judgment JSON files.")
    parser.add_argument(
        "--output-summaries-dir",
        default="summaries_en",
        help="Directory to write translated summaries.",
    )
    parser.add_argument(
        "--output-judgments-dir",
        default="judgments_en",
        help="Directory to write translated judgments.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model to use for translation.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def require_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"Environment variable {key} must be set before running this script.")
    return value


def translate_text(client: OpenAI, model: str, text: str) -> str:
    if not text.strip():
        return text
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful translator. Translate Chinese text into English while preserving the formatting as much as possible."},
            {"role": "user", "content": f"Translate the following text to English:\n\n{text}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def translate_json_preserving_structure(client: OpenAI, model: str, json_text: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful translator. Translate any Chinese text appearing in JSON value fields to English. "
                    "Do not modify keys, numbers, or array ordering. Always return valid JSON."
                ),
            },
            {"role": "user", "content": json_text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def iter_files_with_suffix(directory: Path, suffixes: Iterable[str]) -> Iterable[Path]:
    for suffix in suffixes:
        yield from directory.glob(f"*{suffix}")


def translate_summaries(
    client: OpenAI,
    model: str,
    source_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    translated = 0
    for source_path in sorted(iter_files_with_suffix(source_dir, [".txt"])):
        text = source_path.read_text(encoding="utf-8")
        translated_text = translate_text(client, model, text)
        destination_path = output_dir / source_path.name
        destination_path.write_text(translated_text, encoding="utf-8")
        translated += 1
    logging.info("Translated %d summary files into %s", translated, output_dir)


def translate_judgments(
    client: OpenAI,
    model: str,
    source_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    translated = 0
    for source_path in sorted(iter_files_with_suffix(source_dir, [".json"])):
        json_text = source_path.read_text(encoding="utf-8")
        translated_json_text = translate_json_preserving_structure(client, model, json_text)
        # Validate that we still have valid JSON before writing.
        json.loads(translated_json_text)
        destination_path = output_dir / source_path.name
        destination_path.write_text(translated_json_text, encoding="utf-8")
        translated += 1
    logging.info("Translated %d judgment files into %s", translated, output_dir)


def main() -> None:
    args = parse_args()
    setup_logging()

    summaries_dir = Path(args.summaries_dir)
    judgments_dir = Path(args.judgments_dir)
    if not summaries_dir.exists():
        logging.warning("Summaries directory %s does not exist.", summaries_dir)
    if not judgments_dir.exists():
        logging.warning("Judgments directory %s does not exist.", judgments_dir)

    openai_client = OpenAI(api_key=require_env("OPENAI_API_KEY"))

    if summaries_dir.exists():
        translate_summaries(openai_client, args.model, summaries_dir, Path(args.output_summaries_dir))
    if judgments_dir.exists():
        translate_judgments(openai_client, args.model, judgments_dir, Path(args.output_judgments_dir))


if __name__ == "__main__":
    main()
