import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk, concatenate_datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle



def masked_mean(hidden, mask):
    sum_emb = (hidden * mask.unsqueeze(-1)).sum(dim=1)
    valid = mask.sum(dim=1).clamp(min=1e-9)
    return sum_emb / valid.unsqueeze(-1)

def load_subdataset(subtask):
    dataset_paths = {
        'truthful': '/data/chaojian/Multi-alignment/dataset/alignment_truthful',
        'helpful': '/data/chaojian/Multi-alignment/dataset/ultra_feedback.json',
        'moral': '/data/chaojian/Multi-alignment/dataset/alignment_moral',
        'safety': '/data/chaojian/Multi-alignment/dataset/alignment_pku_safety',
        'stereotype': '/data/chaojian/Multi-alignment/dataset/alignment_stereotype',
        'toxic': '/data/chaojian/Multi-alignment/dataset/alignment_toxic'
    }
    
    if subtask == 'helpful':
        return load_dataset('json', data_files=dataset_paths[subtask])['train']
    return load_from_disk(dataset_paths[subtask])['train']

class TokenizedPromptDataset(Dataset):
    def __init__(self, hf_dataset, subspace_name):
        self.dataset = hf_dataset
        self.subspace_name = subspace_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "label": torch.tensor(self.subspace_name.index(self.dataset[idx]["source"]))
        }


class SimpleClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.linear(x)
    


def train_and_eval_classifier(layer_id, device, NUM_EPOCHS, ):
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    
    with open("/data/chaojian/Multi-alignment/llama2_subspace_hidden_states/layer_hidden_states_with_labels.pkl", "rb") as f:
        saved_dict = pickle.load(f)

    hidden_states = saved_dict[layer_id]["hidden_states"]  # Tensor, shape: (num_samples, hidden_dim)
    labels = saved_dict[layer_id]["labels"]
    # shuffled_labels = labels[torch.randperm(len(labels))]

    X_train, X_val, y_train, y_val = train_test_split(hidden_states, labels, test_size=0.2, random_state=42)

    # 转换为 PyTorch 张量
    # X_train = torch.tensor(X_train, dtype=torch.float32)
    # y_train = torch.tensor(y_train, dtype=torch.long)
    # X_val = torch.tensor(X_val, dtype=torch.float32)
    # y_val = torch.tensor(y_val, dtype=torch.long)

    batch_size = 32
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # X_train, y_train = hidden_states[:120], labels[:120]
    # X_test, y_test = hidden_states[120:], labels[120:]

    clf = SimpleClassifier(hidden_states.shape[1], 6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(clf.parameters(), lr=1e-3)


    for epoch in range(NUM_EPOCHS):
        clf.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            logits = clf(inputs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = running_loss / len(train_dataloader)
        accuracy = correct_predictions / total_samples * 100
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Eval
    clf.eval()
    val_correct_predictions = 0
    val_total_samples = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = clf(inputs)
            _, predicted = torch.max(outputs, 1)

            val_correct_predictions += (predicted == labels).sum().item()
            val_total_samples += labels.size(0)
        
        # total += len(preds)

    # acc = correct / total
    val_accuracy = val_correct_predictions / val_total_samples * 100
    print(f"\nLayer:{layer_id} Validation Accuracy: {val_accuracy:.2f}%\n")
    return val_accuracy

def extract_representation():
    model_name = "../../Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, torch_dtype=torch.bfloat16, device_map="cuda:1").eval()
    tokenizer.pad_token = tokenizer.unk_token


    SUBSPACE_NAMES = [
        'safety', 'toxic', 'helpful', 'moral', 'stereotype', 'truthful', 
    ]


    max_examples_each_subspace = 3000

    datasets = [load_subdataset(name)\
                .shuffle(seed=42)\
                .select(range(max_examples_each_subspace))\
                .map(lambda x: {'source': name})
                for name in SUBSPACE_NAMES]
    
    dataset = concatenate_datasets(datasets)

    def preprocess_function(examples):
        # 定义提示模板
        prompt_input = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""
            
        prompt_no_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
%s

### Response:
"""


        full_prompts = []
        for i in range(len(examples["input"])):
            
            if examples["input"][i]:
                    full_prompts.append(prompt_input % (
                        examples["instruction"][i],
                        examples["input"][i],
                    ) + tokenizer.eos_token)
            else:
                    full_prompts.append(prompt_no_input % (
                        examples["instruction"][i],
                    ) + tokenizer.eos_token)

        inputs = tokenizer(full_prompts, truncation=True, max_length=768, padding="max_length", return_tensors="pt")

        return inputs
    
    dataset = dataset.map(preprocess_function, batched=True)
    dataset = dataset.remove_columns(["instruction", "input", "output"])

    print(dataset[0])

    tokenized_dataset = TokenizedPromptDataset(dataset, SUBSPACE_NAMES)
    dataloader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)

    # layers = [0, 6, 12, 18, 24, 30]

    all_hidden_states = {i: [] for i in range(33)}
    all_labels = []
    with torch.no_grad():
        
        for batch in tqdm(dataloader):
            
            batch = {k: v.to("cuda:1") for k, v in batch.items()}
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states

            for layer in range(len(hidden_states)):
                masked_hidden = masked_mean(hidden_states[layer], batch['attention_mask'])
                all_hidden_states[layer].append(masked_hidden.detach().to('cpu', non_blocking=True))

            all_labels.append(batch["label"].detach().to("cpu", non_blocking=True))
            del outputs, hidden_states,batch
            torch.cuda.empty_cache()
    
    saved_dict = {}
    labels_tensor = torch.cat(all_labels, dim=0)

    for layer in range(33):
        h = torch.cat(all_hidden_states[layer], dim=0)
        saved_dict[layer] = {
            "hidden_states": h,
            "labels": labels_tensor
        }
    with open("/data/chaojian/Multi-alignment/llama2_subspace_hidden_states/layer_hidden_states_with_labels.pkl", "wb") as f:
        pickle.dump(saved_dict, f)

    # torch.save(all_hidden_states, f"/data/chaojian/Multi-alignment/llama2_subspace_hidden_states/subspace_hidden_states.pth")



def main():

    layers = list(range(33))
    layer_accs = []
    for layer in layers:
        acc = train_and_eval_classifier(layer, 'cuda:1', 5)
        layer_accs.append(acc)

    plt.plot(layers, layer_accs, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Classification Accuracy")
    plt.title("Subspace Class Separability per Layer")
    plt.grid(True)
    plt.show()


              
if __name__ == "__main__":
    # extract_representation()
    # train_and_eval_classifier()
    main()


                    
            