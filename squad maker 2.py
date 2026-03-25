import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\Owner\Desktop\projects\training")
CHUNKS_BASE = BASE_DIR / "final chunks"
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
# ✅ UPDATED FUNCTION — Accurate Alias Extraction
# -------------------------------------------------------------------
def parse_data_file(data_path: Path) -> Dict:
    """
    Parses a structured car data file.
    Updated alias extraction logic: supports all cases.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read()

    car_blocks = content.split('---')
    cars = {}

    for block in car_blocks:
        block = block.strip()
        if not block:
            continue

        # Extract main car name
        main_name_match = re.search(r"Car \d+:\s*(.+)", block)
        if not main_name_match:
            continue

        main_name = main_name_match.group(1).strip()
        cars[main_name] = {"aliases": [], "attributes": {}}

        # Extract attributes using existing regex
        for attr_match in re.finditer(r"- ([^:]+): (.+?) \[(\d+),\s*(\d+)\]", block):
            attr_name, value, s, e = attr_match.groups()
            cars[main_name]["attributes"][attr_name.strip()] = {
                "value": value.strip(),
                "span": (int(s), int(e))
            }

        # ✅ Extract aliases using robust logic
        alias_section_match = re.search(r"Car \d+ Aliases:\s*(.*)", block, re.DOTALL)
        if alias_section_match:
            alias_section = alias_section_match.group(1)
            aliases = re.findall(r"^\s*-\s*(.+)", alias_section, re.MULTILINE)
            cars[main_name]["aliases"] = [a.strip() for a in aliases]

    return cars


# -------------------------------------------------------------------
# 🏁 Existing Helper Functions — No Changes
# -------------------------------------------------------------------
def parse_chunk_file(chunk_path: Path) -> Tuple[str, Tuple[int, int]]:
    with open(chunk_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    match = re.search(r"\[(\d+),\s*(\d+)\]\s*$", content)
    if not match:
        raise ValueError(f"Could not find index range in {chunk_path}")

    start_idx, end_idx = map(int, match.groups())

    text = content[:match.start()].strip()
    return text, (start_idx, end_idx)


def word_to_char_index(text: str, target_word_idx: int, global_start: int) -> int:
    words = text.split()
    local_word_idx = target_word_idx - global_start
    if local_word_idx < 0 or local_word_idx >= len(words):
        raise ValueError("Word index out of chunk range.")
    char_index = len(" ".join(words[:local_word_idx])) + (1 if local_word_idx > 0 else 0)
    return char_index


def generate_qas_for_chunk(chunk_text, chunk_range, cars):
    qas = []
    chunk_start, chunk_end = chunk_range

    for car_name, car_info in cars.items():
        aliases = car_info["aliases"] or [car_name]

        for attr_name, attr_data in car_info["attributes"].items():
            if attr_name not in QUESTION_TEMPLATES:
                continue

            value = attr_data["value"]
            ans_start, ans_end = attr_data["span"]

            is_in_chunk = (chunk_start <= ans_start <= chunk_end) or (chunk_start <= ans_end <= chunk_end)

            for alias in aliases:
                for template in QUESTION_TEMPLATES[attr_name]:
                    qid = f"{car_name}_{attr_name}_{alias}_{chunk_start}"
                    question = template.format(car=alias)

                    if is_in_chunk:
                        try:
                            char_start = word_to_char_index(chunk_text, ans_start, chunk_start)
                        except Exception:
                            char_start = 0
                        qas.append({
                            "id": qid,
                            "question": question,
                            "is_impossible": False,
                            "answers": [{"text": value, "answer_start": char_start}]
                        })
                    else:
                        qas.append({
                            "id": qid,
                            "question": question,
                            "is_impossible": True,
                            "answers": []
                        })

    return qas


# -------------------------------------------------------------------
# Main JSON generation pipeline — No Logic Changes
# -------------------------------------------------------------------
def process_transcript_folder(chunk_folder: Path, data_file: Path, output_folder: Path):
    cars = parse_data_file(data_file)

    paragraphs = []
    chunk_files = sorted(chunk_folder.glob("*.txt"), key=lambda p: int(re.search(r"_(\d+)\.txt$", p.name).group(1)))

    for chunk_path in chunk_files:
        chunk_text, chunk_range = parse_chunk_file(chunk_path)
        qas = generate_qas_for_chunk(chunk_text, chunk_range, cars)
        paragraphs.append({"context": chunk_text, "qas": qas})

    squad_data = {
        "data": [
            {"title": chunk_folder.name, "paragraphs": paragraphs}
        ]
    }

    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{chunk_folder.name}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(squad_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Created SQuAD JSON: {output_path}")


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
                print(f"❌ Missing data file for: {transcript_name}")
                continue

            process_transcript_folder(transcript_folder, data_file, output_root)


if __name__ == "__main__":
    main()
