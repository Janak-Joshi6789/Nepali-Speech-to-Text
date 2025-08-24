import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

# Step 1: Load and Prepare the Dataset
data = {
    "No_Space": [
        "समयशक्तिशालीपढ्ननेपालविकास",
        "काठमाडौंशक्तिशालीलाइफ",
        "लाइफपढ्नशक्तिशालीविकास",
        "सम्भावनापढ्नलाइफशक्तिशालीनेपाल",
        "समयविकाससम्भावनासपना",
        "विकाससमयनेपालशिक्षाशक्तिशालीसपना",
        "काठमाडौंशिक्षापढ्नलाइफ",
    ],
    "Correct_Sentence": [
        "समय शक्तिशाली पढ्न नेपाल विकास",
        "काठमाडौं शक्तिशाली लाइफ",
        "लाइफ पढ्न शक्तिशाली विकास",
        "सम्भावना पढ्न लाइफ शक्तिशाली नेपाल",
        "समय विकास सम्भावना सपना",
        "विकास समय नेपाल शिक्षा शक्तिशाली सपना",
        "काठमाडौं शिक्षा पढ्न लाइफ",
    ],
}
df = pd.DataFrame(data)

# Function to generate labels (0 = no space, 1 = space after character)
def generate_labels(no_space, correct_sentence):
    words = correct_sentence.split()
    n = len(no_space)
    # Verify that No_Space is the concatenation of words without spaces
    assert n == sum(len(word) for word in words), "Length mismatch between No_Space and Correct_Sentence"
    labels = [0] * n
    cumsum = 0
    # Set label to 1 after the last character of each word except the last
    for word in words[:-1]:
        cumsum += len(word)
        labels[cumsum - 1] = 1
    return labels

# Apply label generation to the dataframe
df["labels"] = df.apply(lambda row: generate_labels(row["No_Space"], row["Correct_Sentence"]), axis=1)

# Split into train and validation sets (80% train, 20% validation)
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

# Step 2: Load BERT Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)

# Step 3: Define Custom Dataset
class SpaceInsertionDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        no_space = self.df.iloc[idx]["No_Space"]
        labels = self.df.iloc[idx]["labels"]
        
        # Tokenize the input string (splits into individual characters)
        encoding = self.tokenizer(
            no_space,
            add_special_tokens=True,  # Adds [CLS] and [SEP]
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze(0)  # Remove batch dimension
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Adjust labels to match input_ids length: -100 for [CLS] and [SEP]
        labels = [-100] + labels + [-100]
        assert len(input_ids) == len(labels), "Mismatch between input_ids and labels length"
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels),
        }

# Create datasets
train_dataset = SpaceInsertionDataset(train_df, tokenizer)
val_dataset = SpaceInsertionDataset(val_df, tokenizer)

# Step 4: Define Data Collator for Dynamic Padding
data_collator = DataCollatorForTokenClassification(tokenizer)

# Step 5: Set Up Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=10,
)

# Step 6: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Step 7: Train the Model
trainer.train()

# Step 8: Inference Function
def predict_spaces(model, tokenizer, no_space, max_length=512):
    # Tokenize input
    encoding = tokenizer(
        no_space,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding["attention_mask"].to(model.device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).squeeze(0)
    
    # Remove predictions for [CLS] and [SEP]
    predictions = predictions[1:-1]
    
    # Reconstruct string with spaces
    characters = list(no_space)
    result = []
    for char, pred in zip(characters, predictions):
        result.append(char)
        if pred == 1:
            result.append(" ")
    return "".join(result)

# Step 9: Test the Model
test_input = "काठमाडौंशक्तिशालीलाइफ"
predicted_output = predict_spaces(model, tokenizer, test_input)
print(f"Input: {test_input}")
print(f"Predicted: {predicted_output}")