#!/bin/bash

# Define the base directory for configurations and runs
BASE_DIR="/risk1/chengxilong/sentiment-analysis-on-movie-reviews"

# Array of models and their specific configuration file names
declare -A models=(
    [gpt2]="gpt2"
    [gpt2-large]="gpt2-large"
    [bert]="bert"
    [bart-large-mnli]="bart-large-mnli"
    [distilbert-base-uncased]="distilbert-base-uncased"
    [flan-t5-base]="flan-t5-base"
    [twitter-roberta-base-sentiment-latest]="twitter-roberta-base-sentiment-latest"
    [xlnet-base-cased]="xlnet-base-cased"
)

# Loop over each model to setup and run the training
for model in "${!models[@]}"
do
    config_file=${models[$model]}
    log_dir="${BASE_DIR}/runs/${model}-0"
    config_path="${BASE_DIR}/cfg/${config_file}.yaml"
    batch_size=256
    epochs=8
    lr=2e-4

    # Adjust batch size for large models
    if [[ "$model" == "gpt2-large" ]]; then
        batch_size=64
    fi

    # Construct the options string
    options="per_device_train_batch_size=${batch_size}|per_device_eval_batch_size=${batch_size}|num_train_epochs=${epochs}|learning_rate=${lr}"

    # Execute the training command
    CUDA_VISIBLE_DEVICES=4 python ${BASE_DIR}/train.py -l ${log_dir} -mc ${config_path} -o "${options}"
done
