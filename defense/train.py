"""
ERR Two-Stage Training Entry Point.

Stage 1: Train the harm detection head (binary classifier on intermediate hidden states).
Stage 2: Train gated LoRA adapters using the frozen harm head as the gate signal.

Usage:
    # Stage 1 - Harm head training
    deepspeed --num_gpus=8 train.py --config config/err/stage1/err_config_llama.json

    # Stage 2 - LoRA fine-tuning (requires stage 1 head_checkpoint)
    deepspeed --num_gpus=8 train.py --config config/err/stage2/err_config_llama.json
"""
import argparse
import torch
from transformers import AutoTokenizer, TrainingArguments

from err import (
    ERRConfig,
    HarmGatedLlama,
    ERRTrainer,
    ERRDataCollator,
    load_err_dataset,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    config = ERRConfig.from_json(args.config)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = load_err_dataset(
        config.benign_data_path, config.harmful_data_path, tokenizer, config.max_seq_len
    )

    model = HarmGatedLlama(config.base_model_id, config)

    if config.stage == 2:
        if config.head_checkpoint:
            print(f"Loading Stage 1 head: {config.head_checkpoint}")
            state = torch.load(config.head_checkpoint, map_location="cpu", weights_only=True)
            model.harm_head.load_state_dict(state, strict=True)
        else:
            raise ValueError("Stage 2 requires 'head_checkpoint'")

    model.gradient_checkpointing_enable()
    if hasattr(model.base_model, "enable_input_require_grads"):
        model.base_model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.base_model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad
        )

    lr = config.head_lr if config.stage == 1 else config.learning_rate

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        deepspeed=config.deepspeed_config,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        learning_rate=lr,
        num_train_epochs=config.num_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        save_strategy="epoch",
        eval_strategy="epoch",
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=1,
        report_to="wandb",
        remove_unused_columns=False,
    )

    trainer = ERRTrainer(
        err_config=config,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ERRDataCollator(tokenizer),
        tokenizer=tokenizer,
    )

    trainer.train()

    if config.stage == 1:
        model.save_harm_head_only(config.output_dir)
        if args.local_rank in [-1, 0]:
            config.save(config.output_dir)
    elif config.stage == 2:
        model.save_err_weights(config.output_dir)


if __name__ == "__main__":
    main()
