# 🚗 Car Drag Race QA Dataset (SQuAD Format)

---

# 🚨🚨🚨 MORE CODE COMING SOON 🚨🚨🚨

## ⚠️ Additional parts of this project will be uploaded as I get time. Stay tuned for updates!

---

This repository contains a fully custom-built Question Answering (QA) dataset focused on **car drag race data**, designed for fine-tuning BERT-based models (e.g., RoBERTa, BERT, DistilBERT).

The dataset is built from **YouTube transcripts**, processed and transformed into **SQuAD v2.0 format**, and further converted into a **Hugging Face Arrow dataset** for seamless training.

---

## 📌 Project Overview

This project follows an end-to-end pipeline:

1. **Transcript Extraction**

   * YouTube transcripts are extracted using a custom Python script.

2. **Information Parsing**

   * Custom parsers extract structured data such as:

     * Engine size
     * Horsepower
     * Torque
     * Quarter-mile time
     * Other performance metrics

3. **QA Generation**

   * Automatically generates question-answer pairs.
   * Includes:

     * Multiple question variations for the same data
     * Entity-based questions (same car referred using different names)

4. **Answer Indexing**

   * Custom logic is used to compute exact **answer spans (start/end indices)** required for SQuAD format.

5. **SQuAD JSON Creation**

   * Final dataset is structured in **SQuAD v2.0 format**.

6. **Arrow Dataset Conversion**

   * Converted into Hugging Face **Arrow format**, making it directly compatible with:

     * `Trainer API`
     * Pretrained QA models

---

## ✅ Key Features

* 📊 Domain-specific dataset (Car performance & drag races)
* 🤖 Ready for BERT-based QA fine-tuning
* 🔁 Multiple question variations per context
* 🧠 Entity-aware question generation
* 📁 Fully compatible with official SQuAD v2.0 format
* ⚡ Converted to Hugging Face Arrow dataset (plug-and-play)

---

## ⚠️ Important Note

Although this is a **custom dataset**, it has been structured **exactly like the official SQuAD v2.0 dataset**.

👉 This means:

* No compatibility issues
* No special preprocessing required
* Works directly with Hugging Face pipelines

---

## 📂 Repository Structure

```
├── transcripts/              # Raw YouTube transcripts
├── parsers/                  # Custom parsing scripts
├── qa_generation/            # Question generation logic
├── indexing/                 # Answer span indexing scripts
├── squad_json/               # Final SQuAD formatted JSON files
├── arrow_dataset/            # Hugging Face Arrow dataset
├── scripts/                  # Utility scripts
└── README.md
```

---

## 🛠️ What Each Code Module Does (Will Be Expanded)

This repository includes multiple scripts for:

* Transcript processing
* Data extraction
* QA generation
* Answer indexing
* Dataset formatting

📌 **Detailed explanations for each script/module will be added and updated soon.**

---

## 🔄 Future Updates

This project is actively being improved. Upcoming updates include:

* 📖 Detailed documentation for each script
* 🧪 Dataset quality improvements
* 📈 More diverse QA generation
* 🔍 Better entity recognition
* 🧠 Advanced augmentation techniques

---

## 🚀 Usage

Once loaded as an Arrow dataset, it can be directly used with Hugging Face:

```python
from datasets import load_from_disk

dataset = load_from_disk("path_to_arrow_dataset")
```

Then plug into a Trainer for fine-tuning.

---

## 🤝 Contributions

Contributions, ideas, and improvements are welcome!

---

## 📬 Contact

If you’re working on QA systems, NLP, or automotive datasets, feel free to connect or contribute.
