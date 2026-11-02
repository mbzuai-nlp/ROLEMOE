#!/usr/bin/env python3
"""
Inference script for role-playing MoE models
Saves JSON predictions and evaluates using LLM-as-judge
"""

import os
import json
import argparse
import random
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

try:
    import litellm
    LITELLM_AVAILABLE = True
except Exception:
    LITELLM_AVAILABLE = False

from evaluate_rouge import calculate_rouge_scores

torch.set_num_threads(1)


def load_json(file_path: str):
    """Load JSON file"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error loading JSON file {file_path}: JSONDecodeError")
                return []
    return []


def save_json(data: Any, file_path: str):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def setup_model_and_tokenizer(model_name: str, adapter_path: str = None, 
                              torch_dtype=torch.bfloat16):
    """Setup model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if adapter_path:
        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2"
            )
            model.resize_token_embeddings(len(tokenizer))
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch_dtype, 
                attn_implementation="flash_attention_2"
            )

        if not PEFT_AVAILABLE:
            raise RuntimeError("peft not available but adapter_path provided")

        lora_model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
            init_lora_weights=False,
            weights_only=True,
        )
        
        logger.info(f"Inference with MoE.")
        _set_layer_identifiers(lora_model)
        
        for name, module in model.named_modules():
            if hasattr(module, 'set_record_expert_usage'):
                module.set_record_expert_usage(True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lora_model.eval().to(device)
        lora_model = lora_model.to(torch.bfloat16)
        return lora_model, tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)
        model = model.to(torch.bfloat16)
        return model, tokenizer


def _set_layer_identifiers(model):
    """Set layer identifiers for all MoE layers"""
    moe_layer_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'set_layer_identifier'):
            module.set_layer_identifier(name)
            moe_layer_count += 1
    
    logger.info(f"Total MoE layers found: {moe_layer_count}")
    if moe_layer_count == 0:
        logger.warning("No MoE layers found! Check if the model has MoE LoRA layers.")


def _clear_all_layer_histories(model):
    """Clear all layer gate_weights history"""
    for name, module in model.named_modules():
        if hasattr(module, 'gate_weights_history'):
            module.gate_weights_history.clear()


def get_terminators(tokenizer: AutoTokenizer):
    """Get termination token IDs"""
    ids = []
    if getattr(tokenizer, 'eos_token_id', None) is not None:
        ids.append(tokenizer.eos_token_id)
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id not in ids:
            ids.append(eot_id)
    except Exception:
        pass
    return ids or None


def preprocess_function(examples: Dict[str, Any], tokenizer: AutoTokenizer):
    """Preprocess examples for inference"""
    instruction = examples["instruction"]
    role = examples['role']

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

    messages = [
        {"role": "system", "content": "You are a helpful role-playing assistant."},
        {"role": "user", "content": f"{instruction}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ).rstrip()

    model_inputs = tokenizer(
        prompt,
        max_length=1024,
        truncation=True,
        add_special_tokens=False,
    )
    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "role_id": role_id,
        "role_name": role,
        "question": instruction,
    }


def llm_as_judge(question: str, gt_answer: str, predict_answer: str, role_profile: str):
    """Use LLM as judge to evaluate predictions"""
    if not LITELLM_AVAILABLE:
        logger.warning("litellm not available, skipping LLM-as-judge evaluation")
        return None
    
    prompt = f"""
    You will be given a Role Profile, a Question, a Ground Truth, and a Generated Answer.

    You first need to analyze the identity traits given the role profile, question, and ground truth.

    Evaluate the predicted response along the identity consistency with the ground truth. Provide a brief reason for the score.

    ### Role Profile ###
    {role_profile}

    ### Question ###
    {question}

    ### Ground Truth ###
    {gt_answer}

    ### Predicted Response ###
    {predict_answer}

    ### Evaluation Criteria ###  
    You will score the predicted response on a 0-10 scale (higher is better), and provide a brief justification for the score.
    Use the full range of the scale to reflect nuanced differences in quality.

    ### Dimensions to Evaluate ###

    #### 1. Identity Consistency   
    - Definition: The predicted response maintains consistent identity traits across the role profile, question, and ground truth.
    - Consider person/reference (first/third person), claimed abilities/permissions, knowledge/time bounds, tone/style, tool/Memory assumptions, safety boundaries, and avoidance of anthropomorphic claims if disallowed.
    - Penalize contradictions (e.g., claiming web access or personal experiences when not allowed), shifts in persona, or violations of stated limits.
 
    #### 2. Emotional Depth
    - Definition: Is the predicted response emotionally resonant, appropriate to the role, and human-like?
    - Consider emotional range, empathy, tone, and context.
    - Penalize flat, robotic, or inappropriate emotional expressions.

    #### 3. Knowledge Retention
    - Definition: Does the predicted response retain the core knowledge and background information implied by the role profile and ground truth?
    - Consider core facts, established background/lore relevant to the question.
    - Penalize contradictions or lack of knowledge/information.

    #### 4. Semantic Fidelity
    - Definition: Does the predicted response convey the same meaning as the ground truth?
    - Consider core semantic content between ground truth and predicted response.
    - Penalize completely different meaning or lack of semantic content.

    #### 5. Personality & Behavior Alignment
    - Definition: Does the predicted response contain the same personality and behavior as the role profile and ground truth?
    - Consider diction, syntax, rhythm, idioms, catchphrases, decisions, tone, moral/strategic choices.
    - Penalize completely different personality and behavior.

    #### Please provide your evaluation in this format:

    {{
        "identity": {{
            "score": <float>,
            "reason": "Brief justification for the score."
        }},
        "emotional": {{
            "score": <float>,
            "reason": "Brief justification for the score."
        }},
        "knowledge": {{
            "score": <float>,
            "reason": "Brief justification for the score."
        }},
        "similarity": {{
            "score": <float>,
            "reason": "Brief justification for the score."
        }},
        "personality": {{
            "score": <float>,
            "reason": "Brief justification for the score."
        }}
    }}
    """

    try:
        response = litellm.completion(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a role-playing analysis expert. Output ONLY valid minified JSON that matches the requested schema. No extra text."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
        )
        
        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        logger.error(f"Error in LLM-as-judge: {e}")
        return None


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--peft_module", type=str, default=None, 
                       help="Path to LoRA/PEFT adapter")
    parser.add_argument("--evaluation_file", type=str, required=True)
    parser.add_argument("--role_desc_path", type=str, default=None,
                       help="Path to role descriptions JSON file")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--print_text", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_experts", type=int, default=3)
    parser.add_argument("--openai_api_key", type=str, default=None,
                       help="OpenAI API key for LLM-as-judge")
    args = parser.parse_args()

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data = load_json(args.evaluation_file)
    evaluation_file_name = os.path.splitext(os.path.basename(args.evaluation_file))[0]
    
    role_desc_path = args.role_desc_path or "datasets/desc.json"
    role_descs = load_json(role_desc_path)

    if args.peft_module:
        save_json_file = os.path.join(args.peft_module, "results", 
                                     f"{evaluation_file_name}_preds.json")
        os.makedirs(os.path.dirname(save_json_file), exist_ok=True)
        main_results_file = os.path.join(args.peft_module, "results", 
                                        f"{evaluation_file_name}_main_results.json")
        logger.info(f"Saving results to {save_json_file}")
    else:
        save_json_file = f"baseline_results/{args.model_name.split('/')[-1]}/{evaluation_file_name}_preds.json"
        main_results_file = f"baseline_results/{args.model_name.split('/')[-1]}/{evaluation_file_name}_main_results.json"
        os.makedirs(os.path.dirname(save_json_file), exist_ok=True)
        logger.info(f"Saving results to {save_json_file}")

    if os.path.exists(save_json_file):
        logger.info(f"Results file already exists, skipping inference")
        return

    model, tok = setup_model_and_tokenizer(args.model_name, args.peft_module, 
                                          torch.bfloat16)
  
    eos_token_ids = get_terminators(tok)

    for item_idx, item in enumerate(tqdm(data[:])):
        task = item.get('task', 'choice')
        gt_answer = item['output']
        inp = preprocess_function(item, tok)
        input_ids = torch.tensor([inp["input_ids"]], device=model.device)
        attention_mask = torch.tensor([inp["attention_mask"]], device=model.device)
        role_id = torch.tensor([inp["role_id"]], device=model.device)
        role_name = inp["role_name"]

        _clear_all_layer_histories(model)

        with torch.no_grad():
            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                role_id=role_id,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos_token_ids,
                return_dict_in_generate=True,
            )

            out = model.generate(**gen_kwargs)

        in_len = input_ids.shape[-1]
        gen_text = tok.decode(
            out.sequences[0][in_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        item["predict"] = gen_text
        if 'output' in item:
            item["label"] = item['output']

        item['task'] = task
        
        llm_as_judge_result = None
        request_success = False
        
        if role_descs and role_name in role_descs:
            for attempt in range(5):
                try:
                    result_string = llm_as_judge(item['question'], gt_answer, gen_text, role_descs[role_name])
                    if result_string:
                        llm_as_judge_result = json.loads(result_string)
                        if isinstance(llm_as_judge_result, dict) and all(
                            key in llm_as_judge_result for key in 
                            ["identity", "emotional", "knowledge", "similarity", "personality"]
                        ):
                            request_success = True
                            break
                except Exception as e:
                    logger.warning(f"LLM-as-judge attempt {attempt + 1} failed: {e}")
                    continue
        
        if not request_success:
            logger.warning(f"All 5 attempts failed for llm_as_judge, using default values")
            llm_as_judge_result = {
                "identity": {"score": 0.0, "reason": "All retries failed"},
                "emotional": {"score": 0.0, "reason": "All retries failed"},
                "knowledge": {"score": 0.0, "reason": "All retries failed"},
                "similarity": {"score": 0.0, "reason": "All retries failed"},
                "personality": {"score": 0.0, "reason": "All retries failed"}
            }
        
        item['rouge_result'] = calculate_rouge_scores([gen_text], [gt_answer])['rouge_l_mean']
        
        if llm_as_judge_result:
            item['identity'] = float(llm_as_judge_result.get('identity', {}).get('score', 0))
            item['emotional'] = float(llm_as_judge_result.get('emotional', {}).get('score', 0))
            item['knowledge'] = float(llm_as_judge_result.get('knowledge', {}).get('score', 0))
            item['similarity'] = float(llm_as_judge_result.get('similarity', {}).get('score', 0))
            item['personality'] = float(llm_as_judge_result.get('personality', {}).get('score', 0))
        else:
            item['identity'] = 0
            item['emotional'] = 0
            item['knowledge'] = 0
            item['similarity'] = 0
            item['personality'] = 0

        logger.info(f"Rouge: {item['rouge_result']:.4f} | "
                   f"Identity: {item['identity']:.2f} | Emotional: {item['emotional']:.2f} | "
                   f"Knowledge: {item['knowledge']:.2f} | Similarity: {item['similarity']:.2f} | "
                   f"Personality: {item['personality']:.2f}")

    save_json(data, save_json_file)


if __name__ == "__main__":
    main()

