## cd /multi-reft/multi_train

from collections import OrderedDict
import torch
import os

dir1 = './trainer_out_put/Llama2-7b-Nodireft_truthfultoken/checkpoint-7986/intervenable_model'
dir2 = './trainer_out_put/Llama2-7b-Nodireft_toxicitytoken/checkpoint-2817/intervenable_model'
dir3 = './trainer_out_put/Llama2-7b-Nodireft_stereotypetoken_v2/checkpoint-980/intervenable_model'
dir4 = './trainer_out_put/Llama2-7b-Nodireft_safetytoken/checkpoint-564/intervenable_model'
dir5 = './trainer_out_put/Llama2-7b-Nodireft_moraltoken/checkpoint-324/intervenable_model'
dir6 = './trainer_out_put/Llama2-7b-Nodireft_helpfultoken/checkpoint-1458/intervenable_model'

dirs = [dir1, dir2, dir3, dir4, dir5, dir6]
out_dir = './trainer_out_put/Llama2-7b-Nodireft_token_full/intervenable_model'
os.makedirs(out_dir, exist_ok=True)

weight_files = os.listdir(dirs[0])


for weight in weight_files:
    state_dicts = [torch.load(os.path.join(dir, weight)) for dir in dirs]
    merged_state_dict = OrderedDict()

    
    for key in state_dicts[0]:
        if key in ['embed_dim', 'interchange_dim']:
            # 保持这两个键值不变
            merged_state_dict[key] = state_dicts[0][key]
        else:
            # 沿第 0 维拼接权重
            merged_state_dict[key] = torch.cat([sd[key] for sd in state_dicts], dim=0)

    # 保存合并后的权重
    torch.save(merged_state_dict, os.path.join(out_dir, weight))