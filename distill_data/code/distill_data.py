# import torch
# import time
# import json
# from typing import List
# from datasets import load_from_disk
# from openai import OpenAI

# # ======== 配置区域 ========
# API_URL = "https://api.deepseek.com/v1/chat/completions"
# API_KEY = "sk-28fef07857074e77a6134844159f8b08"  # ⚠️ 建议通过环境变量传入更安全

# DATA_PATH = "/data/chaojian/Multi-alignment/dataset/pku-safety"
# OUTPUT_PATH = "Multi-alignment/pku_safety_dataset.jsonl"
# MODEL_NAME = "deepseek-chat"
# MAX_TOKENS = 300
# TEMPERATURE = 0.7
# SLEEP_SECONDS = 1
# # ==========================

# # 模版
# FULL_PROMPT_TEMPLATE = """\
# You are a helpful and responsible AI assistant. Your task is to politely refuse the user's request, \
# especially if it involves harmful, toxic, or unethical behavior. Additionally, provide guidance to steer the user \
# toward a more appropriate course of action.

# User's request:
# "{input}"

# Respond in a professional and respectful tone.
# """

# # 初始化 DeepSeek 客户端
# client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# def call_deepseek(prompt: str, max_tokens: int = MAX_TOKENS) -> str:
#     response = client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=[{'role': 'user', 'content': prompt}],
#         temperature=TEMPERATURE,
#         max_tokens=max_tokens,
#         stream=False
#     )
#     return response.choices[0].message.content.strip()

# def process_input(input_text: str) -> dict:
#     full_prompt = FULL_PROMPT_TEMPLATE.format(input=input_text)
#     full_response = call_deepseek(full_prompt)
#     time.sleep(SLEEP_SECONDS)
#     return {
#         "input": input_text,
#         "full_output": full_response
#     }

# def process_dataset(inputs: List[str], output_path: str):
#     with open(output_path, "w", encoding="utf-8") as fout:
#         for i, input_text in enumerate(inputs):
#             try:
#                 result = process_input(input_text)
#                 fout.write(json.dumps(result, ensure_ascii=False) + "\n")
#                 print(f"[{i+1}] Processed: {input_text[:50]}...")
#             except Exception as e:
#                 print(f"[{i+1}] Failed: {e}")
#                 continue

# def main():
#     print("Loading dataset...")
#     data = load_from_disk(DATA_PATH)
#     train_data = data['train']
#     prompts = train_data['prompt']
#     print(f"Loaded {len(prompts)} prompts.")
    
#     print("Start processing...")
#     process_dataset(prompts, OUTPUT_PATH)
#     print("Done.")

# if __name__ == "__main__":
#     main()

import os
import time
import json
import argparse
from typing import List
from datasets import load_from_disk
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# ======== 配置区域 ========
API_KEY = "sk-e459d7a3a7f743cbbb83d6764380eb6b" # 通过环境变量传入
DATA_PATH = "/data/chaojian/Multi-alignment/dataset/pku-safety"
OUTPUT_PATH = "Multi-alignment/pku_safety_dataset.jsonl"
MODEL_NAME = "deepseek-chat"
MAX_TOKENS = 300
TEMPERATURE = 0.7
SLEEP_SECONDS = 1
# ==========================

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

FULL_PROMPT_TEMPLATE = """\
You are a helpful and responsible AI assistant. Your task is to politely refuse the user's request, \
especially if it involves harmful, toxic, or unethical behavior. Additionally, provide guidance to steer the user \
toward a more appropriate course of action.

User's request:
"{input}"

Respond in a professional and respectful tone.
"""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_deepseek(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {str(e)[:100]}...")
        raise

def process_input(input_text: str) -> dict:
    full_prompt = FULL_PROMPT_TEMPLATE.format(input=input_text)
    full_response = call_deepseek(full_prompt)
    time.sleep(SLEEP_SECONDS)
    return {
        "input": input_text,
        "full_output": full_response.replace("\n", "\\n")  # 转义换行符
    }

def process_dataset(inputs: List[str], output_path: str):
    with open(output_path, "w", encoding="utf-8") as fout:
        for i, input_text in enumerate(inputs):
            try:
                result = process_input(input_text)
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
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--output", type=str, default="distill_output_part.jsonl")
    return parser.parse_args()

def main():
    print("Loading dataset...")
    data = load_from_disk(DATA_PATH)
    prompts = data['train']['prompt'][2092:]
    print(f"Loaded {len(prompts)} prompts.")
    
    print("Processing...")
    args = parse_args()
    all_inputs = prompts[args.start: args.end if args.end > 0 else None]
    output_path = args.output
    process_dataset(all_inputs, output_path)
    
    print("Validating...")
    if validate_jsonl(OUTPUT_PATH):
        print("Validation successful!")
    else:
        print("Validation failed. Check errors.")

if __name__ == "__main__":
    main()