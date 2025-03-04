import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm  # Import tqdm for the progress bar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def classify(df, model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()  # set model to evaluation mode

    texts = df['notes'].tolist()

    # Tokenize the texts in batch
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    # Move inputs to the GPU if available
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Create a TensorDataset and DataLoader for batch processing
    dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size)

    predictions = []
    with torch.no_grad():
        # Wrap dataloader with tqdm to show progress
        for batch in tqdm(dataloader, desc="Inference Progress"):
            input_ids, attention_mask = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1)
            predictions.extend(batch_preds.cpu().numpy())

    df['conflict_related'] = predictions
    df.to_csv('riots_and_protests_annotated.csv', index=False)



if __name__ == '__main__':
    model_dir = "models/finetuned_conflict_classifier"
    df = pd.read_csv('data/riots_and_protests.csv')
    df = df[(df['country'] != 'Palestine') & (df['country'] != 'Israel')]
    classify(df, model_dir)




