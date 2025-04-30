import json
from datasets import load_dataset
import argparse
import os
import time
import numpy as np

import openai
from tqdm import tqdm
from typing import Any
import logging
from typing import List, Dict, Any

import tiktoken
gpt_encoder = tiktoken.get_encoding("cl100k_base")

logging.basicConfig(level=logging.INFO)
logging.getLogger("openai").setLevel(logging.WARNING)  # Reduce OpenAI module logging level
logger = logging.getLogger(__name__)





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_output', type=str, required=True)
    parser.add_argument("--api_model",type=str,default='deepseek-chat')
    parser.add_argument("--api_base",type=str,default='')
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--save_per_examples', type=int, default=5)
    parser.add_argument('--merged_partial_file', type=str, default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size to call OpenAI GPT",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries for API requests and response parsing",
    )

    return parser.parse_args()
    
def gen_prompt(ques, ans1, ans2):

    sys_prompt = "You are a helpful and precise assistant for checking the quality of the answer."
    prompt_template = "[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{criteria}\n\n"
    criteria = """
You are tasked with evaluating which assistant provided a better answer to the given question. 
Consider the following aspects:
- Helpfulness: How well does the answer address the user's question or need?
- Relevance: How relevant is the content to the user's original request?
- Correctness: Is the information accurate and free of mistakes?
- Clarity: Is the answer easy to understand, well-organized, and concise?

After evaluating, you must only output one of the following three options exactly:
- Assistant 1
- Assistant 2
- Tie
Do not explain your choice. Do not provide any additional comments. Only output exactly one of the three options above."""
    prompt = prompt_template.format(
        question=ques, answer_1=ans1, answer_2=ans2, criteria=criteria
    )
    return sys_prompt, prompt

def dispatch_openai_requests(
    messages_list: List[List[Dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    client,
    args
) -> List[str]:
    """Dispatches requests to OpenAI API asynchronously with enhanced error handling.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model (0-2).
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model (0-1).

    Returns:
        List of responses from OpenAI API or error messages.
    """
    # Validate parameters
    if not (0 <= temperature <= 2):
        raise ValueError("temperature must be between 0 and 2")
    if not (0 <= top_p <= 1):
        raise ValueError("top_p must be between 0 and 1")

    # Use client from main program
    

    def create_chat_completion(x):
        max_retries = args.max_retries
        retry_delay = 5
        for attempt in range(max_retries + 1):  # 包含初始尝试和重试次数
            current_delay = retry_delay * (attempt + 1)  # 指数退避
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=x,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return {"error": "Rate limit exceeded"}
            except openai.APIError as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return {"error": f"API Error: {str(e)}"}
            except Exception as e:
                return {"error": f"Unexpected error: {str(e)}"}

    # Create tasks with individual error handling
    return [create_chat_completion(x) for x in messages_list]


def parse_score(review):
    try:
        # 使用正则表达式提取所有数字
        import re
        numbers = re.findall(r"[-+]?\d*\.?\d+", review)
        
        # 验证至少找到两个数字
        if len(numbers) >= 2:
            return [float(numbers[0]), float(numbers[1])]
        
        # 处理带中文数字的情况（例如"3分"）
        cn_numbers = re.findall(r"(\d+)\s*分", review)
        if len(cn_numbers) >= 2:
            return [float(cn_numbers[0]), float(cn_numbers[1])]
            
        # 处理特殊分隔符
        alt_split = re.split(r'[,\t/|]', review.split("\n")[0])
        alt_numbers = [s.strip() for s in alt_split if s.strip().isdigit()]
        if len(alt_numbers) >= 2:
            return [float(alt_numbers[0]), float(alt_numbers[1])]
            
        raise ValueError(f"未找到有效评分: {review[:200]}...")
        
    except Exception as e:
        logger.error(f"评分解析失败: {str(e)}\n原始内容: {review[:500]}...")
        return [-1, -1]


import re
def parse_decision(review):

    try:
        review_clean = review.strip()  # 去除前后空格
        
        # 正则模式，忽略大小写，确保是独立的词
        if re.search(r"\bassistant\s*1\b", review_clean, re.IGNORECASE):
            return "assistant_1"
        elif re.search(r"\bassistant\s*2\b", review_clean, re.IGNORECASE):
            return "assistant_2"
        elif re.search(r"\btie\b", review_clean, re.IGNORECASE):
            return "tie"
        else:
            # 处理只有数字的情况 "1" 或 "2"
            if re.search(r"\b1\b", review_clean):
                return "assistant_1"
            if re.search(r"\b2\b", review_clean):
                return "assistant_2"
        
        raise ValueError(f"未能解析出明确的选择: {review[:200]}...")
        
    except Exception as e:
        logger.error(f"决策解析失败: {str(e)}\n原始内容: {review[:500]}...")
        return "parse_error"

def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list
    

def main():

    args = parse_args()

    reference_output = './dataset/alpaca_eval.json'
    model_output = args.model_output

    # 加载参考输出和模型输出
    with open(reference_output, 'r') as f:
        ref_data = json.load(f)
    with open(model_output, 'r') as f:
        model_data = json.load(f)

    # 提取并合并字段
    if args.merged_partial_file is not None:
        with open(args.merged_partial_file, 'r') as f:
            partial_data = json.load(f)
            merged_data = partial_data.get("data", [])
            print(f"检测到已有部分结果，准备从中间继续...")
            start_idx = len(merged_data)
            print(f"已完成样本数量：{start_idx}")

    else:
        merged_data = []
        for ref_item, model_item in zip(ref_data, model_data):
            merged_data.append({
                "instruction": ref_item["instruction"],
                "model_output": model_item["output"],
                "reference_output": ref_item["output"]
                
            })
        start_idx = 0

    wraped_data = {}
    wraped_data['meta_info'] = {}
    meta_info = wraped_data['meta_info']
    
    
    client = openai.OpenAI(
        api_key=os.environ['DEEPSEEK_API'],
        base_url=args.api_base if args.api_base else None,
        max_retries=0  # Disable OpenAI's default retry logic
    )
    

    if args.max_samples == -1:
        total_len = len(merged_data)
    else:
        total_len = min(args.max_samples, len(merged_data))


    question_idx_list = list(range(total_len))
    
    output_review_file = args.model_output.strip('.json') + f'_reviews_{args.api_model}_test.json'
    partial_file = output_review_file + '_partial.json' if not args.merged_partial_file else args.merged_partial_file
    
    
    
    predictions_all = []
    all_scores = []
    for reverse in range(1): # reverse or not
        message_list = []
        token_len_list = []

        for i in question_idx_list:

            instruction = merged_data[i]['instruction']
            ques = instruction

            if reverse : # reverse = 1, secondly
                ans1 = merged_data[i]['reference_output']
                ans2 = merged_data[i]['model_output']
            else: # reverse = 0, firstly
                ans1 = merged_data[i]['model_output']
                ans2 = merged_data[i]['reference_output']
            sys_prompt, prompt = gen_prompt(ques, ans1, ans2)

            message =[
                        {"role": "system", "content": sys_prompt},
                        {
                            "role": "user",
                            "content": prompt,
                        },
            ]
            message_list.append(message)
            token_len_list.append(len(gpt_encoder.encode(prompt)))

        predictions = []
        scores_list = []


        i = start_idx
        # 上一次保存的的idx的下一个样本的索引
        last_save_idx = start_idx
        wait_base = 10
        retry = 0
        error = 0
        pbar = tqdm(total=len(message_list))
        pbar.update(start_idx)
        batch_size = args.batch_size

        while(i<len(message_list)):
            token_limit_in_current_batch = min(args.max_tokens,4070-max(token_len_list[i:i+batch_size]))
            try:
                batch_predictions = dispatch_openai_requests(
                    messages_list=message_list[i:i+batch_size],
                    model=args.api_model,
                    temperature=0.0,
                    max_tokens=token_limit_in_current_batch,
                    top_p=1.0,
                    client=client,
                    args=args
                )
                predictions += batch_predictions
                retry = 0
                i += batch_size
                wait_base = 10
                pbar.update(batch_size)
                    
                for idx, prediction in enumerate(batch_predictions):
                    real_idx = last_save_idx + idx

                    review = "无法解析模型响应"
                    scores = [-1, -1]
                    try:
                        if isinstance(prediction, dict):
                            if 'error' in prediction:
                                logger.error(f"API请求失败: {prediction['error']}")
                                review = "API请求失败，无法获取评价"
                            else:
                                logger.error(f"无效的响应格式: {str(prediction)[:200]}")
                                review = "无效的响应格式"
                        elif not hasattr(prediction, 'choices'):
                            logger.error(f"响应缺少choices属性: {str(type(prediction))}")
                            review = "响应结构异常"
                        else:
                            review = prediction.choices[0].message.content
                            scores = parse_decision(review)
                    except Exception as e:
                        logger.error(f"处理响应时发生异常: {str(e)}")
                        scores = [-1, -1]
                    review_key = 'review' if not reverse else 'review_reverse'
                    scores_key = 'scores' if not reverse else 'scores_reverse'
                    merged_data[real_idx][review_key] = review
                    merged_data[real_idx][scores_key] = str(scores)
                    scores_list.append(scores)
                
                    wraped_partial_data = {
                        "meta_info": {},  # meta_info 暂时空着
                        "data": merged_data[:i]
                    }
                    with open(partial_file, 'w') as f:
                        json.dump(wraped_partial_data, f, indent=4)
                
                last_save_idx = i
                                      
            except:
                retry += 1
                error += 1
                print("Batch error: ",i, i+batch_size)
                print("retry number: ", retry)
                print("error number: ", error)
                time.sleep(wait_base)
                wait_base = wait_base*2

        pbar.close()
    # all_scores = []
    # for reverse in range(1):
    #     scores_list = []
    #     predictions = predictions_all[reverse]
    #     for idx, prediction in enumerate(predictions):
    #         review = "无法解析模型响应"
    #         scores = [-1, -1]
            
    #         try:
    #             if isinstance(prediction, dict):
    #                 if 'error' in prediction:
    #                     logger.error(f"API请求失败: {prediction['error']}")
    #                     review = "API请求失败，无法获取评价"
    #                 else:
    #                     logger.error(f"无效的响应格式: {str(prediction)[:200]}")
    #                     review = "无效的响应格式"
    #             elif not hasattr(prediction, 'choices'):
    #                 logger.error(f"响应缺少choices属性: {str(type(prediction))}")
    #                 review = "响应结构异常"
    #             else:
    #                 review = prediction.choices[0].message.content
    #                 scores = parse_score(review)
    #         except Exception as e:
    #             logger.error(f"处理响应时发生异常: {str(e)}")
    #             scores = [-1, -1]
    #         review_key = 'review' if not reverse else 'review_reverse'
    #         scores_key = 'scores' if not reverse else 'scores_reverse'
    #         merged_data[idx][review_key] = review
    #         merged_data[idx][scores_key] = str(scores)
    #         scores_list.append(scores)

    print("全部完成，正在统一计算 win_rate 和 avg_scores...")

    all_scores = []
    for entry in merged_data:
        if 'scores_reverse' in entry:
            try:
                scores = eval(entry['scores_reverse'])
                if isinstance(scores, list) and len(scores) == 2:
                    all_scores.append(scores)
            except Exception as e:
                print(f"解析scores_reverse失败，跳过：{entry.get('instruction', '')[:30]}...")
        elif 'scores' in entry:
            try:
                scores = eval(entry['scores'])
                if isinstance(scores, list) and len(scores) == 2:
                    all_scores.append(scores)
            except Exception as e:
                print(f"解析scores失败，跳过：{entry.get('instruction', '')[:30]}...")

    scores_array = np.array(all_scores)
    win_rate = (scores_array[:, 0] >= scores_array[:, 1]).sum() / len(all_scores)
    avg_scores = scores_array.mean(0)

    meta_info['avg_score'] = str(avg_scores.tolist())
    meta_info['win_rate'] = str(win_rate)

    wraped_data['meta_info'] = meta_info
    wraped_data['data'] = merged_data
    
    output_review_file = args.model_output.strip('.json') + f'_reviews_{args.api_model}.json'
    with open(f"{output_review_file}", "w") as f:
        json.dump(wraped_data, f, indent=4)
        pass

    print('Finish:',args.model_output)

if __name__ == "__main__":
    main()

