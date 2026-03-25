import os
import json

folder_path = input("Enter the folder path of squad_jsons: ").strip()

total_qa_pairs = 0

# Walk through all subfolders recursively
for root, _, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(".json"):
            file_path = os.path.join(root, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            file_count = 0

            for article in data.get("data", []):
                for paragraph in article.get("paragraphs", []):
                    for qa in paragraph.get("qas", []):
                        file_count += 1

            print(f"{os.path.relpath(file_path, folder_path)}: {file_count} QA pairs")
            total_qa_pairs += file_count

print("\n--------------------------------")
print(f"✅ Total QA pairs in all subfolders: {total_qa_pairs}")
