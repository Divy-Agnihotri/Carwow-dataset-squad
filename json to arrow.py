import json
import pandas as pd
import pyarrow as pa

def flatten_squad(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for article in data["data"]:
        title = article.get("title", "")

        for paragraph in article["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                rows.append({
                    "id": qa["id"],
                    "title": title,
                    "context": context,
                    "question": qa["question"],
                    "answers": {
                        "text": [a["text"] for a in qa["answers"]],
                        "answer_start": [a["answer_start"] for a in qa["answers"]]
                    }
                })

    return rows




from datasets import Dataset
train_rows = flatten_squad("AMG G63 v V8 Defender Octa DRAG RACE_transcript.json")
print(train_rows)
train_data = Dataset.from_list(train_rows)



print(type(train_data))

arrow_data = train_data._data
print(type(arrow_data))


