import os
import time
import json
import argparse
from typing import List
from datasets import load_from_disk
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# ======== 配置区域 ========
DATA_PATH = "/data/chaojian/Multi-alignment/dataset/crows_pair_stereotype"
OUTPUT_PATH = "Multi-alignment/stereotype_dataset.jsonl"
MODEL_NAME = "qwen-plus-latest"
MAX_TOKENS = 300
TEMPERATURE = 0.7
SLEEP_SECONDS = 1




client = 

FULL_PROMPT_TEMPLATE = """\
You are a helpful, respectful, and ethical assistant. When a user says something toxic, offensive, hateful, or aggressive—such as using slurs, encouraging harm, or promoting hate—you must:
    1. Calmly and firmly reject the toxic behavior or language.
    2. Avoid repeating or reinforcing the harmful content.
    3. Offer an alternative way to express frustration or concern.
    4. Encourage respectful and constructive conversation.
 Respond in a professional and respectful tone.
"""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_deepseek(model_name:str, sys_prompt: str, prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{'role':'system', 'content':sys_prompt},
                      {'role': 'user', 'content': prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {str(e)[:100]}...")
        raise

def process_input(model_name: str, input_text: str) -> dict:
    sys_prompt = FULL_PROMPT_TEMPLATE
    full_response = call_deepseek(model_name=model_name, sys_prompt=sys_prompt, prompt=input_text)
    time.sleep(SLEEP_SECONDS)
    return {
        "input": input_text,
        "full_output": full_response.replace("\n", "\\n")  # 转义换行符
    }

def process_dataset(model_name:str, inputs: List[str], output_path: str):
    with open(output_path, "w", encoding="utf-8") as fout:
        for i, input_text in enumerate(inputs):
            try:
                result = process_input(model_name, input_text)
                json_line = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                fout.write(json_line + "\n")
                fout.flush()
                print(f"[{i+1}] Processed: {input_text[:50]}...")
            except Exception as e:
                print(f"[{i+1}] Error: {str(e)[:100]}...")
                continue

def validate_jsonl(file_path: str) -> bool:
    with open(file_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON at line {i+1}: {str(e)[:100]}...")
                print(f"Problematic content: {line[:200]}...")
                return False
    return True



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="qwen-plus")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--output", type=str, default="distill_output_part.jsonl")
    return parser.parse_args()

def main():
    args = parse_args()
    print("Loading dataset...")
    data = load_from_disk(args.data_path)
    prompts = data['text']
    print(f"Loaded {len(prompts)} prompts.")
    
    print("Processing...")
    
    all_inputs = prompts[args.start: args.end if args.end > 0 else None]
    output_path = args.output
    model_name = args.model_name
    process_dataset(model_name, all_inputs, output_path)
    
    print("Validating...")
    if validate_jsonl(OUTPUT_PATH):
        print("Validation successful!")
    else:
        print("Validation failed. Check errors.")

if __name__ == "__main__":
    main()