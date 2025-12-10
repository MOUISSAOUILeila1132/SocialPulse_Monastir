# =====================================================
# IMPORTS
# =====================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import evaluate
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# =====================================================
# 1. LOAD DATASET
# =====================================================
df = pd.read_csv("Dataset/CombinedDataset.csv")  # mettre le chemin correct

df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(str)

print("Aperçu du dataset :")
print(df.head())

# =====================================================
# 2. DICTIONNAIRE DES CATEGORIES D'ÉMOTIONS
# =====================================================
categorie = {
    "anger": 0,
    "fear": 1,
    "joy": 2,
    "love": 3,
    "none": 4,
    "sadness": 5,
    "sympathy": 6,
    "surprise": 7
}
print("\nDictionnaire des catégories :", categorie)

# Encodage des labels
df["label_id"] = df["label"].map(categorie)
num_labels = len(categorie)
print("\nNombre de classes :", num_labels)

# =====================================================
# 3. CHECK GPU
# =====================================================
if torch.cuda.is_available():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGPU disponible : {gpu_name}\n")
else:
    device = "cpu"
    print("\nAucun GPU détecté. Entraînement sur CPU.\n")

# =====================================================
# 4. SPLIT TRAIN / TEST
# =====================================================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label_id"]
)

# =====================================================
# 5. CREATION DES DATASETS HUGGINGFACE
# =====================================================
train_dataset = Dataset.from_pandas(train_df)
test_dataset  = Dataset.from_pandas(test_df)

# Supprimer les colonnes inutiles
train_dataset = train_dataset.remove_columns(["label"])
test_dataset  = test_dataset.remove_columns(["label"])

# Ajouter la colonne 'labels' en int
train_dataset = train_dataset.add_column("labels", train_df["label_id"].values)
test_dataset  = test_dataset.add_column("labels", test_df["label_id"].values)

# =====================================================
# 6. TOKENIZER + MODEL
# =====================================================
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# =====================================================
# 7. TOKENIZATION FUNCTION
# =====================================================
def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        padding=True,
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset  = test_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =====================================================
# 8. TRAINING ARGUMENTS
# =====================================================
training_args = TrainingArguments(
    output_dir="./bert_emotion_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,   # augmenter si GPU le permet
    per_device_eval_batch_size=16,
    num_train_epochs=10,              # 10 epochs
    weight_decay=0.01,
    logging_steps=50,
    do_eval=True,
    save_strategy="epoch",
    evaluation_strategy="epoch"
)

# =====================================================
# 9. TRAINER
# =====================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# =====================================================
# 10. TRAINING
# =====================================================
print("\n===== DÉBUT DE L'ENTRAÎNEMENT =====")
trainer.train()
print("===== FIN DE L'ENTRAÎNEMENT =====\n")

# =====================================================
# 11. EVALUATION
# =====================================================
print("===== ÉVALUATION DU MODÈLE =====")

# Prédictions train
train_pred = trainer.predict(train_dataset)
train_preds = np.argmax(train_pred.predictions, axis=1)
train_labels = train_pred.label_ids

# Prédictions test
test_pred = trainer.predict(test_dataset)
test_preds = np.argmax(test_pred.predictions, axis=1)
test_labels = test_pred.label_ids

# Accuracy
metric_acc = evaluate.load("accuracy")
train_acc = metric_acc.compute(predictions=train_preds, references=train_labels)["accuracy"]
test_acc  = metric_acc.compute(predictions=test_preds, references=test_labels)["accuracy"]

print(f"Train Accuracy : {train_acc:.4f}")
print(f"Test Accuracy  : {test_acc:.4f}\n")

# Classification report
print("===== Classification Report (Test) =====")
print(classification_report(test_labels, test_preds, target_names=list(categorie.keys())))

# Matrice de confusion
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=list(categorie.keys()), yticklabels=list(categorie.keys()), cmap="Blues")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.title("Matrice de Confusion")
plt.show()

# =====================================================
# 12. SAVE MODEL + TOKENIZER + LABEL ENCODER
# =====================================================
model.save_pretrained("./bert_emotion_model")
tokenizer.save_pretrained("./bert_emotion_model")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(categorie, f)  # on sauvegarde le dictionnaire
print("\nModèle, tokenizer et dictionnaire 'categorie' sauvegardés.")
