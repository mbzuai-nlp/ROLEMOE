#!/usr/bin/env python3
"""
Training script using Role Play MoE LoRA layers
"""

import os
import sys
import torch
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from loguru import logger

# Add peft path
peft_path = os.path.join(os.path.dirname(__file__), '..', 'peft', 'src')
sys.path.insert(0, peft_path)

# Add transformers path
transformers_path = os.path.join(os.path.dirname(__file__), '..', 'transformers', 'src')
sys.path.insert(0, transformers_path)

from peft import LoraConfig, get_peft_model


def setup_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer


def preprocess_function(example, tokenizer):
    instruction = example['instruction']
    role = example['role']
    
    role_id_map = {
        'Gregory House': 0,
        'Rorschach': 1,
        'The Dude': 2,
        'Juno MacGuff': 3,
        'Jeff Spicoli': 4,
        'D_Artagnan': 5,
        'Fred Flintstone': 6,
        'Willie Soke': 7,
        'Gaston': 8,
        'Theodore Twombly': 9,
        'Alvy Singer': 10,
        'Mater': 11,
        'Coriolanus': 12,
        'Karl Childers': 13,
        'Mark Renton': 14, 
    }

    role_id = role_id_map[role]

    output = example['output']

    messages = [
        {"role": "system", "content": "You are a helpful role-playing assistant."},
        {"role": "user", "content": f"{instruction}"},
        {"role": "assistant", "content": output}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    ).rstrip()
    
    model_inputs = tokenizer(
        prompt,
        max_length=256,
        padding="max_length",
        truncation=True,
    )
    
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    
    labels = input_ids[:]
    assistant_start = prompt.find(output)
    if assistant_start == -1:
        raise ValueError("Assistant output not found in the prompt.")
    
    tokens_before_assistant = tokenizer(
        prompt[:assistant_start],
        add_special_tokens=False
    )["input_ids"]
    
    assistant_token_start_idx = len(tokens_before_assistant)
    
    labels[:assistant_token_start_idx] = [-100] * assistant_token_start_idx
    
    padding_start_idx = len(input_ids) - input_ids.count(tokenizer.pad_token_id)
    labels[padding_start_idx:] = [-100] * (len(input_ids) - padding_start_idx)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "role_id": role_id,
    }


class RoleMoETrainer(Trainer):
    """Custom Trainer for Role MoE"""
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step to log additional losses"""
        result = super().training_step(model, inputs, num_items_in_batch)

        if hasattr(model, 'diversity_losses') and model.diversity_losses is not None:
            diversity_loss = model.diversity_losses.item() if hasattr(model.diversity_losses, 'item') else model.diversity_losses
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({
                    "step": self.state.global_step,
                    "diversity_loss": diversity_loss,
                })

        return result

def train(
    train_input_json_file,
    model_name,
    output_dir,
    use_moe,
    role_diversity_loss_coef,
):
    # Check if input file exists
    if not os.path.exists(train_input_json_file):
        raise FileNotFoundError(f"Training file not found: {train_input_json_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {model_name}")
    model, tokenizer = setup_model_and_tokenizer(model_name)

    print(f"Loading dataset from: {train_input_json_file}")
    dataset = Dataset.from_json(train_input_json_file)

    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        remove_columns=dataset.column_names,
        batch_size=8,
    )

    logger.info(f"Creating LoRA config with use_moe={use_moe}")
    
    config_kwargs = {
        "inference_mode": False,
        "init_lora_weights": True,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ['q_proj', 'k_proj', "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "r": 8,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "use_moe": use_moe,
        "num_experts": 3,
        "num_roles": 15,
        "gate_temperature": 1.0,
        "load_balancing_loss_coef": 0,
        "role_diversity_loss_coef": role_diversity_loss_coef,
        "role_diversity_loss_type": "kl",
    }
    
    peft_config = LoraConfig(**config_kwargs)
    lora_model = get_peft_model(model, peft_config)
    lora_model = lora_model.to(torch.bfloat16)
    lora_model.print_trainable_parameters()

    training_args_dict = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "num_train_epochs": 2,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "bf16": True,
        "logging_steps": 1,
        "logging_dir": f"{output_dir}/logs",
        "save_strategy": "no",
        "prediction_loss_only": True,
        "remove_unused_columns": False,
        "optim": "adamw_torch",
        "save_safetensors": False,
        "report_to": ["tensorboard"],
    }

    training_args = TrainingArguments(**training_args_dict)
    trainer = RoleMoETrainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    lora_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model with Role Play MoE LoRA")
    parser.add_argument("--train_input_json_file", type=str, required=True, 
                       help="Path to the training JSON file")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Name or path of the base model")
    parser.add_argument("--use_moe", type=int, default=1,
                       help="Use MoE layers (default: 1)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for the trained model (default: auto-generated)")
    parser.add_argument("--role_diversity_loss_coef", type=float, default=0.0,
                       help="Coefficient for the expert diversity regularization that discourages different experts from co-activating on the same inputs")
    args = parser.parse_args()
    
    print("Role Play MoE LoRA Training")
    print(f"Training file: {args.train_input_json_file}")
    print(f"Model: {args.model_name}")
    print(f"Use MoE: {args.use_moe}")
    print(f"Role diversity loss coef: {args.role_diversity_loss_coef}")
    print("=" * 50)

    train(
        args.train_input_json_file,
        args.model_name,
        args.output_dir,
        args.use_moe,
        args.role_diversity_loss_coef,
    )
