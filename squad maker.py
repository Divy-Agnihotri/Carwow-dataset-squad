import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\Owner\Desktop\projects\training")
CHUNKS_BASE = BASE_DIR / "chunked transcripts"
INFO_BASE = BASE_DIR / "info 2"
OUTPUT_BASE = BASE_DIR / "squad_jsons"

# Standard question templates for each attribute
QUESTION_TEMPLATES = {
    "Engine Type": ["What is the engine type of {car}?"],
    "Engine Capacity": ["What is the engine capacity of {car}?"],
    "Horsepower": ["What is the horsepower of {car}?"],
    "Torque": ["What is the torque of {car}?"],
    "Gearbox": ["What is the gearbox in the {car}?"],
    "Weight": ["What is the weight of {car}?"],
    "Quarter Mile Time": ["What is the quarter mile time of {car}?"]
}


# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------

def parse_chunk_file(chunk_path: Path) -> Tuple[str, Tuple[int, int]]:
    """
    Reads a chunk file and returns:
        - chunk text (context)
        - (start_word, end_word)
    """
    with open(chunk_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # The last line should contain something like [2120,2459]
    match = re.search(r"\[(\d+),\s*(\d+)\]\s*$", content)
    if not match:
        raise ValueError(f"Could not find index range in {chunk_path}")

    start_idx, end_idx = map(int, match.groups())

    # Remove the index line to get the actual text
    text = content[:match.start()].strip()
    return text, (start_idx, end_idx)


def parse_data_file(data_path: Path) -> Dict:
    """
    Parses a structured car data file.
    Returns a dictionary:
    {
        "Car Name": {
            "aliases": [...],
            "attributes": {
                "Horsepower": {"value": "585 horsepower", "span": (1155,1156)},
                ...
            }
        },
        ...
    }
    """
    with open(data_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    cars = {}
    current_car = None
    parsing_aliases = False

    for line in lines:
        if line.startswith("Car "):
            # Example: Car 1: Mercedes AMG G63
            parsing_aliases = False
            current_car = re.sub(r"Car \d+:\s*", "", line).strip()
            cars[current_car] = {"aliases": [], "attributes": {}}

        elif line.startswith("- ") and not parsing_aliases:
            # Attribute line: - Horsepower: 585 horsepower [1155, 1156]
            attr_match = re.match(r"- ([^:]+): (.+?) \[(\d+),\s*(\d+)\]$", line)
            if attr_match:
                attr_name, value, s, e = attr_match.groups()
                cars[current_car]["attributes"][attr_name.strip()] = {
                    "value": value.strip(),
                    "span": (int(s), int(e))
                }

        elif line.endswith("Aliases:"):
            # e.g., Car 1 Aliases:
            parsing_aliases = True

        elif parsing_aliases and line.startswith("- "):
            alias = line[2:].strip()
            cars[current_car]["aliases"].append(alias)

    return cars


def word_to_char_index(text: str, target_word_idx: int, global_start: int) -> int:
    """
    Convert a global word index (e.g., 1155) to a character index within this chunk.
    `global_start` is the first word index of this chunk.
    """
    words = text.split()
    local_word_idx = target_word_idx - global_start
    if local_word_idx < 0 or local_word_idx >= len(words):
        raise ValueError("Word index out of chunk range.")
    # Join up to that word to get char offset
    char_index = len(" ".join(words[:local_word_idx])) + (1 if local_word_idx > 0 else 0)
    return char_index


def generate_qas_for_chunk(chunk_text, chunk_range, cars):
    """
    Generate QA pairs for a given chunk.
    """
    qas = []
    chunk_start, chunk_end = chunk_range

    for car_name, car_info in cars.items():
        aliases = car_info["aliases"] or [car_name]
        for attr_name, attr_data in car_info["attributes"].items():
            if attr_name not in QUESTION_TEMPLATES:
                continue

            value = attr_data["value"]
            ans_start, ans_end = attr_data["span"]

            # Determine if the answer lies in this chunk
            is_in_chunk = (chunk_start <= ans_start <= chunk_end) or (chunk_start <= ans_end <= chunk_end)

            for alias in aliases:
                for template in QUESTION_TEMPLATES[attr_name]:
                    question = template.format(car=alias)

                    if is_in_chunk:
                        # Convert to character index
                        try:
                            char_start = word_to_char_index(chunk_text, ans_start, chunk_start)
                        except Exception:
                            char_start = 0
                        qas.append({
                            "id": f"{car_name}_{attr_name}_{alias}_{chunk_start}",
                            "question": question,
                            "is_impossible": False,
                            "answers": [{"text": value, "answer_start": char_start}]
                        })
                    else:
                        qas.append({
                            "id": f"{car_name}_{attr_name}_{alias}_{chunk_start}",
                            "question": question,
                            "is_impossible": True,
                            "answers": []
                        })
    return qas


# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------
def process_transcript_folder(chunk_folder: Path, data_file: Path, output_folder: Path):
    """Combine one transcript’s chunks + data file into a single SQuAD JSON."""
    cars = parse_data_file(data_file)

    paragraphs = []
    chunk_files = sorted(chunk_folder.glob("*.txt"), key=lambda p: int(re.search(r"_(\d+)\.txt$", p.name).group(1)))

    for chunk_path in chunk_files:
        chunk_text, chunk_range = parse_chunk_file(chunk_path)
        qas = generate_qas_for_chunk(chunk_text, chunk_range, cars)
        paragraphs.append({"context": chunk_text, "qas": qas})

    squad_data = {
        "data": [
            {
                "title": chunk_folder.name,
                "paragraphs": paragraphs
            }
        ]
    }

    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{chunk_folder.name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(squad_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Created: {output_path}")


def main():
    for folder_num in range(1, 6):
        chunk_root = CHUNKS_BASE / str(folder_num)
        info_root = INFO_BASE / str(folder_num)
        output_root = OUTPUT_BASE / str(folder_num)
        if not chunk_root.exists() or not info_root.exists():
            print(f"⚠️ Skipping folder {folder_num} (missing directories)")
            continue

        for transcript_folder in chunk_root.iterdir():
            if not transcript_folder.is_dir():
                continue
            transcript_name = transcript_folder.name
            data_file = info_root / f"{transcript_name}.txt"

            if not data_file.exists():
                print(f"❌ Missing data file for {transcript_name}")
                continue

            process_transcript_folder(transcript_folder, data_file, output_root)


if __name__ == "__main__":
    main()
