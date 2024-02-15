import os
import random
import numpy as np
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from sep_dataset import SEP_Dataset
from dis_model import Dis_Classifier

seed = 0
epochs = 100
batch_size = 128
eval_split = 0.1
eval_log_iterations = 200
checkpoint_iterations = 1000
debug = False

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

if not debug:
    wandb.init(
        project="Typical_Atypical",
        name="sep_wordrep_mfcc_cnn_blstm",
        # notes=notes,
        config={
            "model": "cnn_blstm",
            "seed": seed,
            "batch_size": batch_size,
            "epochs": epochs,
        }
    )
else:
    epochs = 1
    batch_size = 2
    eval_log_iterations = 10

dataset = SEP_Dataset(
    class_a="SoundRep",
    class_b="Fluent"
)
feature_dim = dataset[0][0].shape[1]
train_dataset, eval_dataset = random_split(dataset, [1 - eval_split, eval_split])

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=SEP_Dataset.collate_fn
)

eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=batch_size,
    collate_fn=SEP_Dataset.collate_fn
)

print(f"train_dataloader: {len(train_dataloader)} ({len(train_dataset)}), eval_dataloader: {len(eval_dataloader)} (len{eval_dataset})")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"using device: {device} for training")

model = Dis_Classifier(feature_dim=feature_dim, n_convolutions=3, kernel_size=5, hidden_dim=128)
model.to(device)
model.train()
print(model)

if not debug:
    wandb.define_metric("loss", summary="min")
    wandb.define_metric("loss_eval", summary="min")
    wandb.define_metric("accuracy", summary="max")
    wandb.define_metric("accuracy_eval", summary="max")

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCEWithLogitsLoss()

def get_accuracy(labels_pred, labels):
    labels_pred = F.sigmoid(labels_pred)
    labels_pred = (labels_pred > 0.5).float()
    return torch.eq(labels_pred, labels).float().sum().item() / labels.size(0)

iteration = 0
for epoch in tqdm(range(epochs)):
    for batch in train_dataloader:
        logs = {}

        x, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        logs.update({
            "loss": loss.item(),
            "accuracy": get_accuracy(y_pred, y) * 100
        })
        print(iteration, logs["loss"], logs["accuracy"])
        loss.backward()
        optimizer.step()
        iteration += 1

        if iteration % eval_log_iterations == 0:
            model.eval()
            loss_list_e = []
            accuracy_e = []
            for batch in eval_dataloader:
                x, y = batch[0].to(device), batch[1].to(device)
                with torch.no_grad():
                    y_pred = model(x)
                loss = criterion(y_pred, y)
                loss_list_e += [loss.item()]
                accuracy_e += [get_accuracy(y_pred, y)]
            logs.update({
                "loss_eval": np.mean(loss_list_e),
                "accuracy_eval": np.mean(accuracy_e) * 100
            })
            print("eval:", iteration, logs["loss_eval"], logs["accuracy_eval"])
            model.train()
        
        if iteration % checkpoint_iterations == 0:
            torch.save({
                "state_dict": model.state_dict(),
                "loss": logs["loss"],
                "iteration": iteration,
            }, os.path.join("outputs", f"checkpoint_{iteration}"))

        if not debug:
            wandb.log(logs, step=iteration, commit=True)
