import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from additional_code.data_clean import read_and_clean_data

def compute_metrics_conflict(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "f1": f1}

def compute_metrics_stance(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "f1": f1}


def fine_tune(annotated_df):
    df = annotated_df


    df_training = df[['event_id_cnty','notes', 'conflict_related']]

    def lable_extract(df):
        res = []
        for _, row in df.iterrows():
            lable = [0,0]
            if row['conflict_related'] == 1:
                lable[0] = 1
                if row['pro_palestine'] == 1:
                    lable[1] = 1
                elif row['pro_israel'] == 1:
                    lable[1] = 0
                else:
                    lable[1] = None
            else:
                lable[0] = 0
                lable[1] = None
            res.append(lable)
        return res

    lables = lable_extract(df)
    df_training['related_lable'] = [lable[0] for lable in lables]
    df_training['pro_lable'] = [lable[1] for lable in lables]

    df_related = df_training[df_training['related_lable'] == 1]
    df_related_zeros_sampled = df_training[df_training['related_lable'] == 0].sample(n=len(df_related), random_state=42)
    df_related = pd.concat([df_related, df_related_zeros_sampled]).reset_index(drop=True)
    df_related = df_related.sample(frac=1, random_state=42).reset_index(drop=True)
    raw_conflict_dict = [ (row['notes'], row['related_lable']) for _, row in df_related.iterrows()]

    texts, labels = zip(*raw_conflict_dict)
    raw_conflict_dict = {
        "text": list(texts),
        "conflict_label": list(labels)
    }

    df_pro = df_training[df_training['pro_lable'] == 0]
    df_related_zeros_sampled = df_training[df_training['pro_lable'] == 1].sample(n=len(df_pro), random_state=42)
    df_pro = pd.concat([df_pro, df_related_zeros_sampled]).reset_index(drop=True)
    df_pro = df_pro.sample(frac=1, random_state=42).reset_index(drop=True)
    raw_stance_dict = [ (row['notes'], int(row['pro_lable'])) for _, row in df_pro.iterrows()]

    texts, labels = zip(*raw_stance_dict)
    raw_stance_dict = {
        "text": list(texts),
        "stance_label": list(labels)
    }

    # -------------------------
    # Create Hugging Face Datasets
    # -------------------------
    conflict_dataset = Dataset.from_dict(raw_conflict_dict)
    stance_dataset = Dataset.from_dict(raw_stance_dict)

    # -------------------------
    # Tokenization and Splitting for Conflict Detection Task
    # -------------------------
    model_name = "roberta-large-mnli"
    tokenizer_conflict = AutoTokenizer.from_pretrained(model_name)

    def tokenize_conflict(examples):
        return tokenizer_conflict(examples["text"], truncation=True, padding="max_length", max_length=128)

    # Map tokenization on conflict dataset
    dataset_conflict = conflict_dataset.map(tokenize_conflict, batched=True)
    # Rename 'conflict_label' to 'labels'
    dataset_conflict = dataset_conflict.rename_column("conflict_label", "labels")
    # Set format for PyTorch
    dataset_conflict.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Split into training and evaluation sets (80/20 split)
    split_conflict = dataset_conflict.train_test_split(test_size=0.1, seed=42)
    train_conflict = split_conflict["train"]
    eval_conflict = split_conflict["test"]

    # -------------------------
    # Tokenization and Splitting for Stance Classification Task
    # -------------------------
    tokenizer_stance = AutoTokenizer.from_pretrained(model_name)

    def tokenize_stance(examples):
        return tokenizer_stance(examples["text"], truncation=True, padding="max_length", max_length=128)

    # Map tokenization on stance dataset
    dataset_stance = stance_dataset.map(tokenize_stance, batched=True)
    # Rename 'stance_label' to 'labels'
    dataset_stance = dataset_stance.rename_column("stance_label", "labels")
    # Set format for PyTorch
    dataset_stance.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Split into training and evaluation sets (80/20 split)
    split_stance = dataset_stance.train_test_split(test_size=0.1, seed=42)
    train_stance = split_stance["train"]
    eval_stance = split_stance["test"]

    # -------------------------
    # Model and Trainer Setup for Conflict Classifier
    # -------------------------
    model_conflict = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
    training_args_conflict = TrainingArguments(
        output_dir="./results_conflict",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs_conflict",
        learning_rate=2e-5,
        load_best_model_at_end=True,
    )

    def compute_metrics_conflict(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    trainer_conflict = Trainer(
        model=model_conflict,
        args=training_args_conflict,
        train_dataset=train_conflict,
        eval_dataset=eval_conflict,
        compute_metrics=compute_metrics_conflict,
    )

    # Uncomment the lines below to train and save the conflict classifier
    trainer_conflict.train()
    model_conflict.save_pretrained("./finetuned_conflict_classifier")
    tokenizer_conflict.save_pretrained("./finetuned_conflict_classifier")

    # -------------------------
    # Model and Trainer Setup for Stance Classifier
    # -------------------------
    model_stance = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
    training_args_stance = TrainingArguments(
        output_dir="./results_stance",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs_stance",
        learning_rate=2e-5,
        load_best_model_at_end=True,
    )

    def compute_metrics_stance(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    trainer_stance = Trainer(
        model=model_stance,
        args=training_args_stance,
        train_dataset=train_stance,
        eval_dataset=eval_stance,
        compute_metrics=compute_metrics_stance,
    )

    # Uncomment the lines below to train and save the stance classifier
    trainer_stance.train()
    model_stance.save_pretrained("./finetuned_stance_classifier")
    tokenizer_stance.save_pretrained("./finetuned_stance_classifier")


if __name__ == '__main__':
    path_to_annotated_data = '<enter path to annotated data contain conflict_related, pro_israel, pro_palestine columns>'
    annotated_df = pd.read_csv(path_to_annotated_data)
    fine_tune(annotated_df)