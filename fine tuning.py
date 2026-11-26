import torch
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    TrainingArguments, Trainer, default_data_collator,
)

# ---------- Device ----------
assert torch.cuda.is_available(), "CUDA not available - check your driver / PyTorch build"
device = torch.device("cuda")
print("Using GPU:", torch.cuda.get_device_name(0))

# bf16 needs an Ampere or newer GPU (RTX 30xx/40xx, A100, ...); otherwise use fp16
use_bf16 = torch.cuda.is_bf16_supported()

# ---------- Load Arrow dataset ----------
# Option A: folder created with dataset.save_to_disk(...)
dataset = load_from_disk("carwow_squad")   # Dataset or DatasetDict

# Option B: raw .arrow files
# dataset = load_dataset(
#     "arrow",
#     data_files={"train": "path/train.arrow", "validation": "path/validation.arrow"},
# )

# If you only have a single split, create the validation split yourself:
# dataset = dataset.train_test_split(test_size=0.1)
# dataset = {"train": dataset["train"], "validation": dataset["test"]}

# ---------- Tokenizer ----------
model_name = "deepset/roberta-base-squad2"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

max_length, stride = 384, 128

def preprocess(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")

    start_positions, end_positions = [], []
    for i, offsets in enumerate(offset_mapping):
        answer = examples["answers"][sample_map[i]]
        cls_index = inputs["input_ids"][i].index(tokenizer.cls_token_id)
        sequence_ids = inputs.sequence_ids(i)

        if len(answer["answer_start"]) == 0:          # unanswerable (SQuAD 2.0)
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        ctx_start = sequence_ids.index(1)
        ctx_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        if not (offsets[ctx_start][0] <= start_char and offsets[ctx_end][1] >= end_char):
            start_positions.append(cls_index)         # answer not in this chunk
            end_positions.append(cls_index)
        else:
            idx = ctx_start
            while idx <= ctx_end and offsets[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = ctx_end
            while idx >= ctx_start and offsets[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

train_ds = dataset["train"].map(
    preprocess, batched=True, remove_columns=dataset["train"].column_names
)
eval_ds = dataset["validation"].map(
    preprocess, batched=True, remove_columns=dataset["validation"].column_names
)

# ---------- Model ----------


# ---------- Training ----------
args = TrainingArguments(
    output_dir="roberta-qa-finetuned",
    num_train_epochs=2,               # 1-3 is usually enough on top of an already-tuned model
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=1e-5,               # lower than the 3e-5 used for base BERT
    weight_decay=0.01,
    warmup_steps=0.1,
    bf16=use_bf16,
    fp16=not use_bf16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=50,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    data_collator=default_data_collator,
)

# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last N transformer layers (RoBERTa-base has 12)
N = 4
for layer in model.roberta.encoder.layer[-N:]:
    for param in layer.parameters():
        param.requires_grad = True

# Always train the QA head
for param in model.qa_outputs.parameters():
    param.requires_grad = True

# Check how many parameters are trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

trainer.train()
trainer.save_model("bert-qa-finetuned/final")