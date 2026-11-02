#!/usr/bin/env python3
"""
Training script for role-playing MoE LoRA unlearning
"""

import os
import sys
import torch
import argparse
import yaml
import shutil
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import PeftModel, PeftConfig
import datasets
from loguru import logger

peft_path = os.path.join(os.path.dirname(__file__), '..', 'peft', 'src')
if os.path.exists(peft_path):
    sys.path.insert(0, peft_path)

transformers_path = os.path.join(os.path.dirname(__file__), '..', 'transformers', 'src')
if os.path.exists(transformers_path):
    sys.path.insert(0, transformers_path)

MAIN_PATH = "datasets/json_format/rolebench_finetune_unlearning"


def setup_model_and_tokenizer(model_id):
    """Setup model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer


def configure_trainable_parameters(model):
    """Configure trainable parameters for the model"""
    for n, p in model.named_parameters():
        if "lora_role_gating" in n or ("lora_experts_A" in n) or ("lora_experts_B" in n):
            p.requires_grad = True


class RoleMoEUnlearningTrainer(Trainer):
    """Custom Trainer for Role MoE unlearning"""
    
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('loss_type')
        self.alpha = kwargs.pop('alpha', 1.0)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss based on loss type (GA or GD)"""
        if self.loss_type == "ga":
            forget_inputs, retain_inputs, _ = inputs
            input_ids, labels, attention_mask, role_ids = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask, role_id=role_ids)
            forget_loss = outputs.loss * -1
            loss = forget_loss
            
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({
                    "step": self.state.global_step,
                    "forget_loss": forget_loss.item(),
                })

        elif self.loss_type == "gradDiff":
            forget_inputs, retain_inputs, _ = inputs
            input_ids, labels, attention_mask, role_ids = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask, role_id=role_ids)
            forget_loss = outputs.loss * -1

            retain_input_ids, retain_labels, retain_attention_mask, retain_role_ids = retain_inputs
            retain_outputs = model(retain_input_ids, labels=retain_labels, 
                                 attention_mask=retain_attention_mask, role_id=retain_role_ids)
            retain_loss = retain_outputs.loss
            loss = forget_loss + self.alpha * retain_loss
            
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({
                    "step": self.state.global_step,
                    "forget_loss": forget_loss.item(),
                    "retain_loss": retain_loss.item(),
                })
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}. Only 'ga' and 'gradDiff' are supported.")

        if not return_outputs:
            outputs = None

        return (loss, outputs) if return_outputs else loss


class RoleMoEDataset(Dataset):
    """Dataset class for Role MoE training"""
    
    def __init__(self, tokenizer, max_length=256, data_path=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = data_path or MAIN_PATH

        retention_path = os.path.join(self.data_path, "retention_set_1300.json")
        forget_path = os.path.join(self.data_path, "forget_set_200.json")
        
        self.retention_data = datasets.load_dataset('json', data_files=retention_path)['train']
        self.forget_data = datasets.load_dataset('json', data_files=forget_path)['train']
        
        idk_path = os.path.join(self.data_path, "idontknow.jsonl")
        with open(idk_path, 'r', encoding='utf-8') as f:
            self.idk_data = [line.strip() for line in f if line.strip()]
        
        self.split1, self.split2, self.split3 = "forget", "retain", "idk"

        self.role_id_map = {
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

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in [self.split1, self.split2, self.split3]:
            if data_type == "retain":
                data = self.retention_data
                retain_idx = (idx + torch.randint(0, len(self.retention_data), (1,)).item()) % len(self.retention_data)
                sample = data[retain_idx]
            elif data_type == "forget":
                data = self.forget_data
                sample = data[idx]
            elif data_type == "idk":
                data = self.forget_data
                rand_pos = torch.randint(0, len(self.idk_data), (1,)).item()
                answer = self.idk_data[rand_pos].strip()
                sample = data[idx]
                sample['output'] = answer

            instruction = sample['instruction']
            role = sample['role']
            output = sample['output']
            
            role_id = self.role_id_map[role]
            
            messages = [
                {"role": "system", "content": "You are a helpful role-playing assistant."},
                {"role": "user", "content": f"{instruction}"},
                {"role": "assistant", "content": output}
            ]
            
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            ).rstrip()
            
            model_inputs = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
            
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]
            
            labels = input_ids[:]
            assistant_start = prompt.find(output)
            if assistant_start == -1:
                raise ValueError("Assistant output not found in prompt.")
            
            tokens_before_assistant = self.tokenizer(
                prompt[:assistant_start],
                add_special_tokens=False
            )["input_ids"]
            
            assistant_token_start_idx = len(tokens_before_assistant)
            
            labels[:assistant_token_start_idx] = [-100] * assistant_token_start_idx
            
            padding_start_idx = len(input_ids) - input_ids.count(self.tokenizer.pad_token_id)
            labels[padding_start_idx:] = [-100] * (len(input_ids) - padding_start_idx)
            
            result = {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
                "role_id": torch.tensor(role_id),
            }
            
            rets.append(result)
        
        return rets


def role_moe_data_collator(samples):
    """Custom collator for Role MoE dataset"""
    forget_samples = [sample[0] for sample in samples]
    retain_samples = [sample[1] for sample in samples]
    idk_samples = [sample[2] for sample in samples]
    
    rets = []
    
    for data_type in ["forget", "retain", "idk"]:
        if data_type == "forget":
            data = forget_samples
        elif data_type == "retain":
            data = retain_samples
        elif data_type == "idk":
            data = idk_samples
        
        input_ids = [s["input_ids"] for s in data]
        attention_mask = [s["attention_mask"] for s in data]
        labels = [s["labels"] for s in data]
        role_ids = [s["role_id"] for s in data]
        
        rets.append((torch.stack(input_ids), torch.stack(labels), 
                    torch.stack(attention_mask), torch.stack(role_ids)))
    
    return rets


def train(
    model_name,
    output_dir,
    pretrained_model_path=None,
    unlearn_learning_rate=1e-5,
    epochs=10,
    use_moe=1,
    loss_type="ga",
    alpha=1.0,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    per_device_train_batch_size=None,
    gradient_accumulation_steps=None,
): 
    """Main training function"""
    os.makedirs(output_dir, exist_ok=True)

    adapter_config = PeftConfig.from_pretrained(pretrained_model_path)

    adapter_config.experts_orthogonal_loss_coef = 0
    adapter_config.role_diversity_loss_coef = 0

    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    lora_model = PeftModel.from_pretrained(
        model,
        pretrained_model_path,
        config=adapter_config,
        is_trainable=True,
        init_lora_weights=False,
        weights_only=True
    )
    lora_model = lora_model.to(torch.bfloat16)

    dataset = RoleMoEDataset(tokenizer=tokenizer, max_length=256)

    if use_moe == 1:
        configure_trainable_parameters(lora_model)

    lora_model.print_trainable_parameters()

    training_args_dict = {
        "output_dir": output_dir,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": unlearn_learning_rate,
        "num_train_epochs": epochs,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": max_grad_norm,
        "weight_decay": 0.01,
        "warmup_steps": 2,
        "bf16": True,
        "logging_steps": 1,
        "logging_dir": f"{output_dir}/logs",
        "save_strategy": "no",
        "save_total_limit": 1,
        "remove_unused_columns": False,
        "optim": "paged_adamw_32bit",
        "save_safetensors": False,
    }
    
    training_args = TrainingArguments(**training_args_dict)
    
    trainer = RoleMoEUnlearningTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=role_moe_data_collator,
        loss_type=loss_type,
        alpha=alpha,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    lora_model.save_pretrained(output_dir)

    print("Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)
    
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unlearn model using Role MoE LoRA")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Base model name or path")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for trained model")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                       help="Path to pretrained PEFT model for continuing training")
    parser.add_argument("--use_moe", type=int, default=1,
                       help="Use MoE (default: 1)")
    parser.add_argument("--config_file", type=str, default=None,
                       help="YAML configuration file path")
    args = parser.parse_args()

    if args.config_file:
        config_path = args.config_file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("No config file provided")

    unlearn_learning_rate = float(config.get("learning_rate"))
    output_dir = args.output_dir

    if os.path.exists(output_dir):
        logger.info(f"Output directory {output_dir} already exists, removing it first")
        shutil.rmtree(output_dir)

    epochs = config.get("epochs")
    alpha = config.get("alpha", 1.0)
    max_grad_norm = config.get("max_grad_norm")
    warmup_ratio = config.get("warmup_ratio")
    per_device_train_batch_size = config.get("per_device_train_batch_size", 1)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 16)
    loss_type = config.get("loss_type")

    if loss_type not in ["ga", "gradDiff"]:
        raise ValueError(f"Unsupported loss type: {loss_type}. Only 'ga' and 'gradDiff' are supported.")

    logger.info("Role-playing MoE LoRA training")
    logger.info("=" * 50)
    logger.info(f"Pretrained model path: {args.pretrained_model_path}")
    logger.info(f"Unlearning learning rate: {unlearn_learning_rate}")
    logger.info(f"Training epochs: {epochs}")
    logger.info(f"Loss type: {loss_type}")
    logger.info(f"Alpha: {alpha}")
    logger.info(f"Max gradient norm: {max_grad_norm}")
    logger.info(f"Warmup ratio: {warmup_ratio}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 50)

    train(
        args.model_name,
        output_dir,
        args.pretrained_model_path,
        unlearn_learning_rate,
        epochs,
        args.use_moe,
        loss_type,
        alpha=alpha,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

