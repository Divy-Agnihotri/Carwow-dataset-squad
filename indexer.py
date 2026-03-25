import os
import re
from difflib import SequenceMatcher

def normalize_numbers(text):
    return re.sub(r'(?<=\d),(?=\d)', '', text)

def fuzzy_match(a, b):
    """Return similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def calculate_word_indices(text, target_text, threshold=0.85):
    """
    Given text and target_text, find word indices (1-based) where target_text appears.
    Uses exact match first; falls back to fuzzy matching if exact fails.
    """
    text = normalize_numbers(text)
    target_text = normalize_numbers(target_text)

    words = text.split()
    target_words = target_text.split()
    n = len(words)
    m = len(target_words)

    best_match = (None, 0.0)  # (indices tuple, similarity score)

    for i in range(n - m + 1):
        segment = " ".join(words[i:i + m])
        if words[i:i + m] == target_words:
            return (i + 1, i + m)
        else:
            sim = fuzzy_match(segment, target_text)
            if sim > best_match[1]:
                best_match = ((i + 1, i + m), sim)

    if best_match[1] >= threshold:
        print(f"[FUZZY MATCH] '{target_text}' matched with similarity {best_match[1]:.2f}")
        return best_match[0]

    return None


def process_transcript_and_data(transcript, data):
    """
    Updates the data file's indices based on the transcript text.
    """
    transcript_text = transcript.replace('\n', ' ')
    data_lines = data.splitlines()
    updated_data = []

    for line in data_lines:
        bracket_match = re.search(r'\[(\d+),\s*(\d+)\]', line)

        if bracket_match:
            match_start = bracket_match.start()
            before_bracket = line[:match_start].strip()

            if ": " in before_bracket:
                target_text = before_bracket.split(": ")[-1].strip()
            else:
                target_text = before_bracket.strip()

            result = calculate_word_indices(transcript_text, target_text)

            if result is not None:
                start_idx, end_idx = result
                new_line = re.sub(r'\[\d+,\s*\d+\]', f'[{start_idx}, {end_idx}]', line)
                updated_data.append(new_line)
            else:
                print(f"[WARNING] Could not find: '{target_text}' in transcript.")
                updated_data.append(line)
        else:
            updated_data.append(line)

    return '\n'.join(updated_data)


def process_folders(transcript_folder, data_folder):
    """
    Process all subfolders and files, updating data files' indices to match transcript files.
    """

    if not os.path.exists(transcript_folder):
        print(f"[ERROR] Transcript folder not found: {transcript_folder}")
        return
    if not os.path.exists(data_folder):
        print(f"[ERROR] Data folder not found: {data_folder}")
        return

    for subfolder in os.listdir(transcript_folder):
        transcript_subfolder = os.path.join(transcript_folder, subfolder)
        data_subfolder = os.path.join(data_folder, subfolder)

        if not os.path.isdir(transcript_subfolder):
            continue
        if not os.path.isdir(data_subfolder):
            print(f"[WARNING] Missing matching folder in data: {data_subfolder}")
            continue

        print(f"\n📂 Processing subfolder: {subfolder}")

        for filename in os.listdir(transcript_subfolder):
            if not filename.lower().endswith(".txt"):
                continue

            transcript_path = os.path.join(transcript_subfolder, filename)
            data_path = os.path.join(data_subfolder, filename)

            if not os.path.exists(data_path):
                print(f"[WARNING] No matching data file for: {filename}")
                continue

            print(f"   📝 Processing: {filename}")

            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_content = f.read()
            with open(data_path, 'r', encoding='utf-8') as f:
                data_content = f.read()

            updated_data = process_transcript_and_data(transcript_content, data_content)

            with open(data_path, 'w', encoding='utf-8') as f:
                f.write(updated_data)

            print(f"   ✅ Updated {filename}")

    print("\n✅ All matching files processed successfully.")


if __name__ == "__main__":
    # Put your actual folder paths here, using raw strings or escaped backslashes
    transcript_folder = r"C:\Users\Owner\Desktop\projects\training\basic transcript"
    data_folder = r"C:\Users\Owner\Desktop\projects\training\info 2"

    process_folders(transcript_folder, data_folder)
