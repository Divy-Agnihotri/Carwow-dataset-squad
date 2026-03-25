import os
import re
from difflib import SequenceMatcher

def fuzzy(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def extract_phrase(line):
    """Extract the phrase after colon and before [indices]."""
    try:
        before_bracket = line.split("[")[0]
        phrase = before_bracket.split(":")[1].strip()
        return phrase
    except:
        return None

def fix_phrases(transcript_text, data_text, threshold=0.85):
    updated_lines = []
    transcript_words = transcript_text.split()

    for line in data_text.splitlines():
        if "[" in line and "]" in line:  # Spec lines only
            phrase = extract_phrase(line)
            if not phrase:
                updated_lines.append(line)
                continue

            phrase_words = phrase.split()
            m = len(phrase_words)

            best_sim = 0.0
            best_segment = None
            found_exact = False

            for i in range(len(transcript_words) - m + 1):
                segment = " ".join(transcript_words[i:i + m])
                if segment.lower() == phrase.lower():
                    found_exact = True
                    break

                sim = fuzzy(segment, phrase)
                if sim > best_sim:
                    best_sim = sim
                    best_segment = segment

            if found_exact:
                updated_lines.append(line)
            else:
                if best_segment and best_sim >= threshold:
                    corrected = best_segment
                    new_line = line.replace(phrase, corrected)
                    updated_lines.append(new_line)
                    print(f"[FIXED] '{phrase}' → '{corrected}' (sim={best_sim:.2f})")
                else:
                    updated_lines.append(line)
                    print(f"[NO MATCH] '{phrase}'")
        else:
            updated_lines.append(line)

    return "\n".join(updated_lines)

def process_all(unchunked_root, data_root):
    for section in os.listdir(unchunked_root):
        sec_path = os.path.join(unchunked_root, section)
        if not os.path.isdir(sec_path):
            continue

        print(f"\n📂 Processing Section: {section}")
        data_sec = os.path.join(data_root, section)

        for file in os.listdir(sec_path):
            if not file.lower().endswith(".txt"):
                continue

            transcript_file = os.path.join(sec_path, file)
            data_file = os.path.join(data_sec, file)

            if not os.path.exists(data_file):
                print(f"[SKIPPED] Missing data file for {file}")
                continue

            print(f"   📝 Checking: {file}")

            with open(transcript_file, 'r', encoding='utf-8') as tf:
                transcript_text = tf.read().replace("\n", " ")

            with open(data_file, 'r', encoding='utf-8') as df:
                data_text = df.read()

            updated_text = fix_phrases(transcript_text, data_text)

            with open(data_file, 'w', encoding='utf-8') as df:
                df.write(updated_text)

            print(f"   ✅ Updated: {file}")

    print("\n🎯 Phrase correction complete (Unchunked Matching)!")

if __name__ == "__main__":
    unchunked_root = input("Enter path to BASIC TRANSCRIPT (unchunked): ").strip()
    data_root = input("Enter path to DATA (info 2): ").strip()
    process_all(unchunked_root, data_root)
