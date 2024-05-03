#!/bin/bash

# 设置目标目录
BASE_DIR="/risk1/chengxilong/sentiment-analysis-on-movie-reviews/weights"

# 定义模型和对应的文件夹名称
declare -A models=(
    ["twitter-roberta-base-sentiment-latest"]="twitter-roberta-base-sentiment-latest"
    ["gpt2-large"]="gpt2-large"
    ["distilbert-base-uncased"]="distilbert-base-uncased"
    ["flan-t5-base"]="flan-t5-base"
    ["bart-large-mnli"]="bart-large-mnli"
    ["xlnet-base-cased"]="xlnet-base-cased"
)

# 主执行循环
for model in "${!models[@]}"
do
    model_dir="$BASE_DIR/${models[$model]}"
    echo "Checking directory: $model_dir"
    # 检查目录是否存在，不存在则创建
    if [ ! -d "$model_dir" ]; then
        echo "Directory does not exist. Creating now"
        mkdir -p "$model_dir"
    else
        echo "Directory exists."
    fi
    # 执行下载命令
    /risk1/chengxilong/hfd.sh $model --tool aria2c -x 4 --local-dir "$model_dir"
done
