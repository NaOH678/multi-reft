import torch
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk, concatenate_datasets

class SimpleClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.linear(x)
    

def train_and_eval_classifier(layer_id, device, NUM_EPOCHS, saved_dict, input_dim=4096, output_dim=6):
    
    # X_train, y_train = hidden_states[:960].to(device), labels[:960].to(device)
    # X_test, y_test = hidden_states[960:].to(device), labels[960:].to(device)
    hidden_states = saved_dict[layer_id]["hidden_states"]  # Tensor, shape: (num_samples, hidden_dim)
    labels = saved_dict[layer_id]["labels"]
    # shuffled_labels = labels[torch.randperm(len(labels))]
    input_dim = hidden_states.shape[1]
    output_dim = torch.unique(labels).shape[0]

    X_train, X_val, y_train, y_val = train_test_split(hidden_states, labels, test_size=0.2, random_state=42)

   
    # batchsize 过大也不行。整个训练集直接训练模型会过拟合
    batch_size = 64
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    clf = SimpleClassifier(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-2)


    for epoch in range(NUM_EPOCHS):
        clf.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = clf(inputs)
            loss = criterion(logits, labels)
            
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


if __name__== "main":

    