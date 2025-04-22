import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer,TrainingArguments, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model

model_path = '/data/chaojian/Qwen-1.5b'




def train():

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    # padding_side 是否需要调整？

    ds = load_dataset('tatsu-lab/alpaca')
    shuffle_dataset = ds['train'].shuffle(seed=42).train_test_split(test_size=0.2)
    selected_dataset_train = shuffle_dataset['train'].select(range(10000))
    selected_dataset_eval = shuffle_dataset['test'].select(range(2000))

    def preprocess_function(examples):
        
        model_inputs = tokenizer(f"Instruction:" + examples['instruction']+ "\nInput:" + examples['input']+'\nResponse',
                                  padding='max_length', max_length=128, truncation=True)
        
        

        labels = tokenizer(examples['output'], padding='max_length', max_length=128, truncation=True)['input_ids']
        labels = [-100 if token_id == tokenizer.pad_token_id else token_id for token_id in labels ]
        model_inputs['labels'] = labels

        return model_inputs
    
    data_collector = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = selected_dataset_train.map(preprocess_function, batch_size=256,remove_columns=["instruction", "input", "output", "text"])
    eval_dataset = selected_dataset_eval.map(preprocess_function, batch_size=256,remove_columns=["instruction", "input", "output", "text"])

    

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=['q_proj','v_proj'],
        layers_to_transform=[10,12,14,16,18],
        lora_dropout=0.1,  # Dropout概率
        bias="none",  # 是否训练偏置
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./qwen_alpaca",
        learning_rate=3e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        fp16=True,  # 启用混合精度训练
        logging_dir="./qwen_alpaca_logs",
        logging_steps=10,
        eval_steps=20,
        evaluation_strategy="steps" ,     # 需要显示设置
        save_strategy="steps",  # 或者 "epoch"
        save_steps=50,         # 每隔多少步保存一次
        deepspeed='/data/chaojian/MoE/baselline/configs/ds_zero0.json'
        
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collector
    )

    trainer.train()


if __name__ == "__main__":
    train()










