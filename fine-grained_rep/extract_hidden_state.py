import torch
import transformers
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tqdm import tqdm
import random
import gc

ds_binarized = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")


model_path = '/data/chaojian/Llama-2-7b-chat-hf'
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='cuda:0', output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.padding_side = "left"
tokenizer.pad_token = "<pad>"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")


formatted_texts = [tokenizer.apply_chat_template(conv, tokenize=False) for conv in ds_binarized['train']['chosen']]

samples_texts = random.sample(formatted_texts, 1000)
dataset = Dataset.from_dict({"text": samples_texts})
def preprocess_function(examples):
    return tokenizer(examples["text"], padding=True, max_length=930, truncation=True)

dataset = dataset.map(preprocess_function, batched=True)
dataset = dataset.remove_columns(["text"])
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

class ModelWithHiddenStates(AutoModelForCausalLM):
    def __init__(self):
        super().__init__()
        self.hidden_states = None

    def forward(self, input_ids, **kwargs):
        outputs = super().forward(input_ids, **kwargs)

        # print(outputs.hidden_states)
    
        return outputs.hidden_states         
    # 对于每一个batch，输出的一个元组，元组长度为33（llama2的层数），
    # 元组中的每一个元素是一个batch在每一层的hidden state，（batch， seq_length， hidden_size）


model = ModelWithHiddenStates.from_pretrained(model_path, torch_dtype=torch.float16, output_hidden_states=True)


ds_model = deepspeed.init_inference(
    model,
    mp_size=1,     # >1 表示模型并行，把模型分割到多张卡；=1表示每张卡都加载模型  
    dtype=torch.float16,
    replace_with_kernel_inject=True
)


# batches = [formatted_texts[i:i+batch_size] for i in range(0, len(samples_texts), batch_size)]
all_hidden_states = {i:[] for i in range(33)}      # num of layers in Llama2

# 提前用dataset_map进行tokenize，用dataloader加载数据，总用时少了40min

for i, batch in enumerate(tqdm(dataloader, desc="Extracting hidden states")):
    # inputs = tokenizer(batch, return_tensors="pt", padding='max_length', max_length=1233, truncation=True).to("cuda")
    batch = {k: v.to("cuda") for k, v in batch.items()}
    with torch.no_grad():
        outputs = ds_model(**batch)
        # print(hidden_states.shape)
    for layer, hidden_state in enumerate(outputs.hidden_states):
        all_hidden_states[layer].append(hidden_state[:,-1, :].detach().cpu())

    del outputs, batch
    # if i % 10 == 0:
    #     # print(f"{i} batches processed")
    gc.collect()
    torch.cuda.empty_cache()

for layer, hidden_states in all_hidden_states.items():
    all_hidden_states[layer] = torch.cat(hidden_states, dim=0)

torch.save(all_hidden_states, "/data/chaojian/Multi-alignment/llama2_preference_hidden_states/llama2_hidden_states_prefered.pth")

    

