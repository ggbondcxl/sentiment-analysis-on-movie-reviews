import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, load_dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from torch.utils.data import DataLoader
import re
import os.path as path
from collections import defaultdict
from argparse import ArgumentParser
import json
import yaml
import socket
import os
import os.path as path
import shutil
from datetime import datetime

# 创建解析器
parser = ArgumentParser()
parser.add_argument('--model-config', '-mc', required=True)
parser.add_argument('--log-dir', '-l')
parser.add_argument('--override', '-o', default='')

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def get_config(config_path):
    with open(config_path, 'r') as f:
        new_config = yaml.full_load(f)
    config = {}
    if 'include' in new_config:
        include_config = get_config(new_config['include'])
        config.update(include_config)
        del new_config['include']
    config.update(new_config)
    return config

def main():
    if torch.cuda.is_available():
        print(f'Running on {socket.gethostname()} | {torch.cuda.device_count()}x {torch.cuda.get_device_name()}')
    else:
        print(f'Running on {socket.gethostname()} | CPU')
    args = parser.parse_args()

    # Load config
    config = get_config(args.model_config)
    
    # Override options
    """ 
    When here = config and you do here = here[key] in a loop, you're actually modifying the reference to here to point to a deeper structure in the config dictionary.
    Since dictionaries are mutable objects in Python, when you change here, you change config.
    Since here is a reference to config, any changes you make to here will also be reflected in config.
    """
    """ for option in args.override.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                here[key] = {}
            here = here[key]
        if keys[-1] not in here:
            print(f'Warning: {address} is not defined in config file.')
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader) """
    for option in args.override.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                here[key] = {}
            here = here[key]
        
        # 尝试解析为适当的数据类型
        if keys[-1] == 'learning_rate':
            # 学习率应该是浮点数
            parsed_value = float(value)
        elif keys[-1] in ['num_train_epochs', 'per_device_train_batch_size', 'per_device_eval_batch_size']:
            # 这些值应该是整数
            parsed_value = int(value)
        else:
            # 对于其他所有值，使用yaml.load来猜测类型
            parsed_value = yaml.load(value, Loader=yaml.FullLoader)
        
        if keys[-1] not in here:
            print(f'Warning: {address} is not defined in config file.')
        here[keys[-1]] = parsed_value
    
    # Prevent overwriting
    config['log_dir'] = args.log_dir
    config_save_path = path.join(config['log_dir'], 'config.yaml')
    try:
        # Try to open config file to bypass NFS cache
        with open(config_save_path, 'r') as f:
            f.read(1)
            config_exists = True
    except FileNotFoundError:
        config_exists = False

    if config_exists:
        print(f'WARNING: {args.log_dir} already exists. Skipping...')
        exit(0) 

    # Save config
    os.makedirs(config['log_dir'], mode=0o755, exist_ok=True)
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f'Config saved to {config_save_path}')

    # Save code
    code_dir = path.join(config['log_dir'], 'code_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    script_name = os.path.basename(__file__)
    shutil.copy2(__file__, os.path.join(config['log_dir'], script_name)) 
    print(f'Code saved to {code_dir}')
    
    train_test(config)

def train_test(config):

    mps_device = torch.device("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
    # model = AutoModelForSequenceClassification.from_pretrained(config['model_path'], num_labels=config['num_labels'])
    if "facebook/bart-large-mnli" or "cardiffnlp/twitter-roberta-base-sentiment-latest" or "facebook/bart-large-mnli" in config['tokenizer']:
        # 加载模型，忽略不匹配的尺寸，并设置新的分类头数量
        model = AutoModelForSequenceClassification.from_pretrained(
            config['model_path'],
            num_labels=config['num_labels'],
            ignore_mismatched_sizes=True  # 忽略预训练模型与新模型间的尺寸不匹配
        )
    else:
        # 对于其他情况，正常加载模型并设置分类头的数量
        model = AutoModelForSequenceClassification.from_pretrained(
            config['model_path'],
            num_labels=config['num_labels']
        )
    model = model.to(mps_device)

    pad_token = '[PAD]'
    if 'pad_token' in config:
        tokenizer.add_special_tokens({'pad_token': pad_token})
        model.resize_token_embeddings(len(tokenizer))
        pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
        tokenizer.pad_token_id = pad_token_id
        model.config.pad_token_id = pad_token_id
    
    df = pd.read_csv(config['train_path'], sep='\t')
    train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df.Sentiment.values)


    # Preprocess training data
    train_df['label'] = train_df['Sentiment']
    valid_df['label'] = valid_df['Sentiment']

    # Convert DataFrame to Dataset and apply preprocessing
    data_train = Dataset.from_pandas(train_df)
    data_valid = Dataset.from_pandas(valid_df)
    data_train = data_train.map(lambda examples: preprocess(tokenizer, examples), batched=True, remove_columns=list(train_df.columns.difference(['Phrase', 'label'])))
    data_valid = data_valid.map(lambda examples: preprocess(tokenizer, examples), batched=True, remove_columns=list(valid_df.columns.difference(['Phrase', 'label'])))

    data_train = Dataset.from_dict({"input_ids": data_train['input_ids'], "attention_mask": data_train['attention_mask'], "labels": data_train['labels']})
    data_valid = Dataset.from_dict({"input_ids": data_valid['input_ids'], "attention_mask": data_valid['attention_mask'], "labels": data_valid['labels']})
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(config['log_dir'], 'results'),
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        warmup_steps= config['warmup_steps'],
        learning_rate=config['learning_rate'],
        logging_dir=os.path.join(config['log_dir'], 'logs'),
        logging_steps=config['logging_steps'],
        evaluation_strategy=config['evaluation_strategy'],
        save_strategy=config['save_strategy'],
        load_best_model_at_end=config['load_best_model_at_end'],
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
    test_df = pd.read_csv(config['test_path'], sep='\t')
    test_df.fillna('NA', inplace=True)  # Handling missing values

    # Preprocess test data to fit the model
    test_dataset = Dataset.from_pandas(test_df)
    test_dataset = test_dataset.map(lambda examples: tokenizer(examples['Phrase'], truncation=True, padding='max_length', max_length=config['max_length']), batched=True)
    test_dataset = test_dataset.remove_columns(['Phrase', 'SentenceId'])  # Remove unnecessary columns

    # Predict using the trained model
    outputs= trainer.predict(test_dataset)
    predictions = outputs.predictions
    if isinstance(predictions, tuple):
        print("Logits tuple contains", len(predictions), "elements")
        for i, prediction_part in enumerate(predictions):
            print(f"Shape of logits part {i}:", prediction_part.shape)
            if prediction_part.ndim == 2:  # 查找二维的 logits 部分
                predictions = prediction_part
                print(f"Using logits part {i} for calculations.")
                break

    # 确认我们有正确的二维 logits 数组
    if predictions.ndim != 2:
        raise ValueError("Expected 2D logits array for calculations.")
    preds = np.argmax(predictions, axis=-1)

    # Output results to CSV
    test_df['Sentiment'] = preds
    submit_df = test_df[['PhraseId', 'Sentiment']]
    submit_path = os.path.join(config['log_dir'], 'submit.csv')
    submit_df.to_csv(submit_path, index=False)

    print("Submission file 'submit.csv' created.")

def preprocess(tokenizer, examples):
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

    if isinstance(logits, tuple):
        print("Logits tuple contains", len(logits), "elements")
        for i, logit_part in enumerate(logits):
            print(f"Shape of logits part {i}:", logit_part.shape)
            if logit_part.ndim == 2:  # 查找二维的 logits 部分
                logits = logit_part
                print(f"Using logits part {i} for calculations.")
                break

    # 确认我们有正确的二维 logits 数组
    if logits.ndim != 2:
        raise ValueError("Expected 2D logits array for calculations.")
    
    predictions = np.argmax(logits, axis=-1)
    f1_micro_average = f1_score(y_true=labels, y_pred=predictions, average='micro')
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    return {'f1': f1_micro_average, 'accuracy': accuracy}

if __name__ == '__main__':
    main()