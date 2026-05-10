# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)

from torch.optim import AdamW

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize


# -----------------------
# DEVICE
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("D:/UNIK/6sem/NNfSPT/7_BERT_classification/data/1.csv")

# если нужно — раскомментируй чистку
# df["class"] = df["class"].astype(str).str.strip().str.lower()

sentences = df["comment"].astype(str).values
labels = df["class"].astype(int).values

print("Samples:", len(df))
print(df["class"].value_counts())


# -----------------------
# TOKENIZER
# -----------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

MAX_LEN = 64

input_ids = np.zeros((len(sentences), MAX_LEN))

for i, s in enumerate(sentences):
    enc = tokenizer.encode(
        s,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True
    )
    input_ids[i] = enc

input_ids = input_ids.astype(int)


# -----------------------
# ATTENTION MASKS
# -----------------------
attention_masks = np.array([[int(id_ > 0) for id_ in seq] for seq in input_ids])


# -----------------------
# TRAIN / VAL SPLIT (95% / 5%)
# -----------------------
train_inputs, val_inputs, train_labels, val_labels, train_masks, val_masks = train_test_split(
    input_ids,
    labels,
    attention_masks,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

print("Train size:", len(train_inputs))
print("Val size:", len(val_inputs))

# -----------------------
# DATASETS
# -----------------------
train_dataset = TensorDataset(
    torch.tensor(train_inputs),
    torch.tensor(train_masks),
    torch.tensor(train_labels)
)

val_dataset = TensorDataset(
    torch.tensor(val_inputs),
    torch.tensor(val_masks),
    torch.tensor(val_labels)
)


# -----------------------
# DATALOADERS
# -----------------------
batch_size = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=SequentialSampler(val_dataset))


# -----------------------
# MODEL (3 CLASSES)
# -----------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

model.to(device)


# -----------------------
# OPTIMIZER
# -----------------------
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

epochs = 2
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


# -----------------------
# SEED
# -----------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# -----------------------
# ACCURACY FUNCTION
# -----------------------
def flat_accuracy(preds, labels):
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    return np.sum(preds == labels) / len(labels)


# -----------------------
# TRAINING
# -----------------------
loss_values = []

for epoch in range(epochs):

    print(f"\nEpoch {epoch+1}/{epochs}")
    t0 = time.time()
    total_loss = 0

    model.train()

    for step, batch in enumerate(train_loader):

        b_input_ids = batch[0].to(device).long()
        b_input_mask = batch[1].to(device).long()
        b_labels = batch[2].to(device).long()

        model.zero_grad()

        outputs = model(
            b_input_ids,
            attention_mask=b_input_mask,
            labels=b_labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(train_loader)
    loss_values.append(avg_loss)

    print("Avg loss:", avg_loss)
    print("Time:", time.time() - t0)


    # -----------------------
    # VALIDATION
    # -----------------------
    model.eval()

    eval_acc = 0
    nb_steps = 0

    for batch in val_loader:

        b_input_ids = batch[0].to(device).long()
        b_input_mask = batch[1].to(device).long()
        b_labels = batch[2].to(device).long()

        with torch.no_grad():
            outputs = model(
                b_input_ids,
                attention_mask=b_input_mask
            )

        logits = outputs.logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()

        eval_acc += flat_accuracy(logits, label_ids)
        nb_steps += 1

    print("Validation accuracy:", eval_acc / nb_steps)


# -----------------------
# LOSS PLOT
# -----------------------
plt.plot(loss_values)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# =======================
# FINAL EVALUATION
# =======================

model.eval()

all_preds = []
all_probs = []
all_true = []

for batch in val_loader:

    b_input_ids = batch[0].to(device).long()
    b_input_mask = batch[1].to(device).long()
    b_labels = batch[2].to(device).long()

    with torch.no_grad():
        outputs = model(
            b_input_ids,
            attention_mask=b_input_mask
        )

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(logits, dim=1)

    all_preds.extend(preds.cpu().numpy())
    all_probs.extend(probs.cpu().numpy())
    all_true.extend(b_labels.cpu().numpy())


all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_true = np.array(all_true)


# -----------------------
# REPORT
# -----------------------
print("\nFinal Accuracy:", (all_preds == all_true).mean())

print("\nClassification Report:")
print(classification_report(
    all_true,
    all_preds,
    target_names=["любовь", "смерть", "дружба"]
))


# -----------------------
# CONFUSION MATRIX
# -----------------------
cm = confusion_matrix(all_true, all_preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["любовь", "смерть", "дружба"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix (3 classes)")
plt.show()


# -----------------------
# ROC CURVE
# -----------------------
n_classes = 3
y_true_bin = label_binarize(all_true, classes=[0, 1, 2])

class_names = ["любовь", "смерть", "дружба"]

plt.figure()

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve (3 classes)")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()


# -----------------------
# SAVE MODEL
# -----------------------
model.save_pretrained("./model_3class")
tokenizer.save_pretrained("./model_3class")