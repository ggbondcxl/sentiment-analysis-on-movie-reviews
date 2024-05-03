import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, load_dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from torch.utils.data import DataLoader
import re

from collections import defaultdict

def preprocess(examples):
    processed_examples = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for idx, phrase in enumerate(examples['Phrase']):
        phrase = phrase.lower()
        phrase = re.sub(r'[^\w\s]', '', phrase)
        tokenized_example = tokenizer(phrase, truncation=True, padding='max_length', max_length=30)
        processed_examples['input_ids'].append(tokenized_example['input_ids'])
        processed_examples['attention_mask'].append(tokenized_example['attention_mask'])
        processed_examples['labels'].append(examples['Sentiment'][idx])
    
    return processed_examples




# Compute metrics function
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    f1_micro_average = f1_score(y_true=labels, y_pred=predictions, average='micro')
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    return {'f1': f1_micro_average, 'accuracy': accuracy}

# Paths to datasets
train_path = '/dataset/chengxilong/sentiment-analysis-on-movie-reviews/train.tsv'
test_path = '/dataset/chengxilong/sentiment-analysis-on-movie-reviews/test.tsv'

# Load and preprocess training data
df = pd.read_csv(train_path, sep='\t')
train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df.Sentiment.values)

# Tokenizer and model setup
model_path = "/risk1/chengxilong/sentiment-analysis-on-movie-reviews/weights/gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Add special tokens if needed and resize token embeddings
model = GPT2ForSequenceClassification.from_pretrained(model_path, num_labels=5)
pad_token = '[PAD]'
tokenizer.add_special_tokens({'pad_token': pad_token})
model.resize_token_embeddings(len(tokenizer))

# Now set the padding token ID for the tokenizer and model configuration
pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
tokenizer.pad_token_id = pad_token_id
model.config.pad_token_id = pad_token_id

# Preprocess training data
train_df['label'] = train_df['Sentiment']
valid_df['label'] = valid_df['Sentiment']

""" # Check a sample from the training data before tokenization
print("Sample from training data before tokenization:")
print(train_df.head())

# Check a sample from the validation data before tokenization
print("\nSample from validation data before tokenization:")
print(valid_df.head()) """

# Convert DataFrame to Dataset and apply preprocessing
data_train = Dataset.from_pandas(train_df)
data_train = data_train.map(preprocess, batched=True, remove_columns=list(train_df.columns.difference(['Phrase', 'label'])))

data_valid = Dataset.from_pandas(valid_df)
data_valid = data_valid.map(preprocess, batched=True, remove_columns=list(valid_df.columns.difference(['Phrase', 'label'])))

data_train = Dataset.from_dict({"input_ids": data_train['input_ids'], "attention_mask": data_train['attention_mask'], "labels": data_train['labels']})
data_valid = Dataset.from_dict({"input_ids": data_valid['input_ids'], "attention_mask": data_valid['attention_mask'], "labels": data_valid['labels']})
""" # Check a sample from training data after tokenization and preprocessing
print("Sample from training data after tokenization and preprocessing:")
print(data_train)

# Check a sample from validation data after tokenization and preprocessing
print("\nSample from validation data after tokenization and preprocessing:")
print(data_valid[0]) """
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
# Training arguments
training_args = TrainingArguments(
    output_dir='/risk1/chengxilong/sentiment-analysis-on-movie-reviews/results',
    num_train_epochs=3,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    warmup_steps=100,
    learning_rate=2e-4,
    logging_dir='/risk1/chengxilong/sentiment-analysis-on-movie-reviews/logs',
    logging_steps=500,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=data_train,            # training dataset
    eval_dataset=data_valid,             # evaluation dataset
    compute_metrics=compute_metrics,     # the function we defined earlier for computing metrics
    data_collator=data_collator  # our data collator to pad the inputs
)

# Train the model
trainer.train()

# Load and preprocess test data
test_df = pd.read_csv(test_path, sep='\t')
test_df.fillna('NA', inplace=True)  # Handling missing values

# Preprocess test data to fit the model
test_dataset = Dataset.from_pandas(test_df)
test_dataset = test_dataset.map(lambda examples: tokenizer(examples['Phrase'], truncation=True, padding='max_length', max_length=30), batched=True)
test_dataset = test_dataset.remove_columns(['Phrase', 'SentenceId'])  # Remove unnecessary columns

# Predict using the trained model
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)

# Output results to CSV
test_df['Sentiment'] = preds
submit_df = test_df[['PhraseId', 'Sentiment']]
submit_df.to_csv('/risk1/chengxilong/sentiment-analysis-on-movie-reviews/submit.csv', index=False)

print("Submission file 'submit.csv' created.")

