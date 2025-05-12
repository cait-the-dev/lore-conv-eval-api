from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def _encode(example, tokenizer):
    context = example["conversation"]
    answer = example["answer_text"]
    start_char = context.find(answer)

    enc = tokenizer(
        f"What evidence shows the user's {example['facet'].lower()}?",
        context,
        truncation=True,
        max_length=384,
        padding="max_length",
        return_offsets_mapping=True,
    )

    offsets = enc.pop("offset_mapping")

    if start_char == -1 or not answer:
        start_idx = end_idx = 0
    else:
        start_idx = end_idx = 0
        for idx, (s, e) in enumerate(offsets):
            if s <= start_char < e:
                start_idx = idx
            if s < start_char + len(answer) <= e:
                end_idx = idx
                break

    enc.update({"start_positions": start_idx, "end_positions": end_idx})
    return enc


def train(csv_path: str, out_dir: str) -> None:
    df = pd.read_csv(csv_path)
    ds = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    ds = ds.map(
        lambda ex: _encode(ex, tokenizer),
        remove_columns=list(df.columns),
        desc="tokenising",
    )

    model = AutoModelForQuestionAnswering.from_pretrained("roberta-base")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=3e-5,
        fp16=(device == "cuda"),
        logging_steps=50,
        save_strategy="epoch",
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    Trainer(
        model=model, args=args, train_dataset=ds, data_collator=data_collator
    ).train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"✅ Saved fine-tuned model to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="models/roberta_multitask")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    train(args.csv, args.out)
