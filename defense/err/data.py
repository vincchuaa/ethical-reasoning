import math
import torch
from typing import Dict
from transformers import PreTrainedTokenizer
from datasets import load_dataset, concatenate_datasets


def normalize_conversation_format(example: Dict) -> Dict:
    if "messages" in example:
        messages = example.get("messages")

        if messages is None or len(messages) == 0:
            raise ValueError("Example has empty or None messages field")

        validated_messages = []
        for msg in messages:
            if msg is None:
                continue
            role = msg.get("role", "user")
            content = msg.get("content")
            if content is None:
                content = ""
            validated_messages.append({"role": role, "content": str(content)})

        if len(validated_messages) == 0:
            raise ValueError("Example has no valid messages after validation")

        return {
            "messages": validated_messages,
            "harm_label": example.get("harm_label", 0.0),
        }

    if "conversations" in example:
        messages = []
        for conv in example["conversations"]:
            role_map = {
                "human": "user",
                "user": "user",
                "gpt": "assistant",
                "assistant": "assistant",
                "system": "system",
            }
            role = role_map.get(conv.get("from", "").lower(), "user")
            content = conv.get("value", conv.get("content", ""))
            messages.append({"role": role, "content": content})

        normalized = {"messages": messages}

        if "metadata" in example:
            metadata = example["metadata"]
            if "success" in metadata:
                normalized["harm_label"] = 0.0 if metadata["success"] else 1.0
            elif "harm_label" in metadata:
                normalized["harm_label"] = metadata["harm_label"]

        if "harm_label" in example:
            normalized["harm_label"] = example["harm_label"]
        elif "harm_label" not in normalized:
            normalized["harm_label"] = 0.0

        return normalized

    raise ValueError(
        f"Example must have either 'messages' or 'conversations' field. "
        f"Got keys: {list(example.keys())}"
    )


def format_conversation(
    example: Dict, tokenizer: PreTrainedTokenizer, max_length: int
) -> Dict:
    """
    Format conversation using robust prompt-length approach.

    Tokenizes the prompt-only version to determine where the assistant response
    starts, then unmasks everything after that length for training.
    """
    example = normalize_conversation_format(example)

    messages = example["messages"]
    harm_label = example.get("harm_label", 0.0)

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    input_ids = tokenized["input_ids"]
    labels = [-100] * len(input_ids)

    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    prompt_tokenized = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    prompt_len = len(prompt_tokenized["input_ids"])

    pad_id = tokenizer.pad_token_id
    for i in range(prompt_len, len(input_ids)):
        if input_ids[i] != pad_id:
            labels[i] = input_ids[i]

    unmasked_count = sum(1 for l in labels if l != -100)
    if unmasked_count == 0:
        print(
            f"[WARNING] No unmasked tokens! prompt_len={prompt_len}, total={len(input_ids)}"
        )

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
        "harm_label": harm_label,
    }


def is_valid_example(example):
    """Check if an example has valid messages structure."""
    if "messages" in example:
        messages = example.get("messages")
        if messages is None or len(messages) == 0:
            return False
        for msg in messages:
            if msg is not None and msg.get("content"):
                return True
        return False

    if "conversations" in example:
        conversations = example.get("conversations")
        if conversations is None or len(conversations) == 0:
            return False
        for conv in conversations:
            if conv is not None and (conv.get("value") or conv.get("content")):
                return True
        return False

    return False


def load_err_dataset(benign_path, harmful_path, tokenizer, max_length, seed=42):
    ds_benign = load_dataset("json", data_files=benign_path, split="train")
    ds_harmful = load_dataset("json", data_files=harmful_path, split="train")

    ds_benign = ds_benign.filter(is_valid_example, num_proc=4)
    ds_harmful = ds_harmful.filter(is_valid_example, num_proc=4)

    if "harm_label" not in ds_benign.column_names:
        ds_benign = ds_benign.map(lambda x: {"harm_label": 0.0})
    if "harm_label" not in ds_harmful.column_names:
        ds_harmful = ds_harmful.map(lambda x: {"harm_label": 1.0})

    n_benign = len(ds_benign)
    n_harmful = len(ds_harmful)

    if n_benign > n_harmful:
        repeat_factor = math.ceil(n_benign / n_harmful)
        ds_harmful = concatenate_datasets([ds_harmful] * repeat_factor).select(
            range(n_benign)
        )
    elif n_harmful > n_benign:
        repeat_factor = math.ceil(n_harmful / n_benign)
        ds_benign = concatenate_datasets([ds_benign] * repeat_factor).select(
            range(n_harmful)
        )

    combined = concatenate_datasets([ds_benign, ds_harmful]).shuffle(seed=seed)
    cols = [c for c in combined.column_names if c != "harm_label"]

    processed = combined.map(
        lambda x: format_conversation(x, tokenizer, max_length),
        remove_columns=cols,
        num_proc=4,
    )
    splits = processed.train_test_split(test_size=0.1, seed=seed)
    return splits["train"], splits["test"]


class ERRDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        return {
            "input_ids": torch.tensor(
                [f["input_ids"] for f in features], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [f["attention_mask"] for f in features], dtype=torch.long
            ),
            "labels": torch.tensor(
                [f["labels"] for f in features], dtype=torch.long
            ),
            "harm_labels": torch.tensor(
                [f["harm_label"] for f in features], dtype=torch.float
            ),
        }
