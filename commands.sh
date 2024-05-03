gpt2
CUDA_VISIBLE_DEVICES=4 python /risk1/chengxilong/sentiment-analysis-on-movie-reviews/train.py -l /risk1/chengxilong/sentiment-analysis-on-movie-reviews/runs/gpt2-0 -mc /risk1/chengxilong/sentiment-analysis-on-movie-reviews/cfg/gpt2.yaml -o "per_device_train_batch_size=256|per_device_eval_batch_size=256|num_train_epochs=3|learning_rate=2e-4"

gpt2-large
CUDA_VISIBLE_DEVICES=4 python /risk1/chengxilong/sentiment-analysis-on-movie-reviews/train.py -l /risk1/chengxilong/sentiment-analysis-on-movie-reviews/runs/gpt2-large-0 -mc /risk1/chengxilong/sentiment-analysis-on-movie-reviews/cfg/gpt2-large.yaml -o "per_device_train_batch_size=64|per_device_eval_batch_size=64|num_train_epochs=3|learning_rate=2e-4"

bert
CUDA_VISIBLE_DEVICES=4 python /risk1/chengxilong/sentiment-analysis-on-movie-reviews/train.py -l /risk1/chengxilong/sentiment-analysis-on-movie-reviews/runs/bert-0 -mc /risk1/chengxilong/sentiment-analysis-on-movie-reviews/cfg/bert.yaml -o "per_device_train_batch_size=256|per_device_eval_batch_size=256|num_train_epochs=3|learning_rate=2e-4"

bart-large-mnli
CUDA_VISIBLE_DEVICES=4 python /risk1/chengxilong/sentiment-analysis-on-movie-reviews/train.py -l /risk1/chengxilong/sentiment-analysis-on-movie-reviews/runs/bart-large-mnli-0 -mc /risk1/chengxilong/sentiment-analysis-on-movie-reviews/cfg/bart-large-mnli.yaml -o "per_device_train_batch_size=128|per_device_eval_batch_size=128|num_train_epochs=3|learning_rate=2e-4"

distilbert-base-uncased
CUDA_VISIBLE_DEVICES=4 python /risk1/chengxilong/sentiment-analysis-on-movie-reviews/train.py -l /risk1/chengxilong/sentiment-analysis-on-movie-reviews/runs/distilbert-base-uncased-0 -mc /risk1/chengxilong/sentiment-analysis-on-movie-reviews/cfg/distilbert-base-uncased.yaml -o "per_device_train_batch_size=256|per_device_eval_batch_size=256|num_train_epochs=3|learning_rate=2e-4"

flan-t5-base
CUDA_VISIBLE_DEVICES=4 python /risk1/chengxilong/sentiment-analysis-on-movie-reviews/train.py -l /risk1/chengxilong/sentiment-analysis-on-movie-reviews/runs/flan-t5-base-0 -mc /risk1/chengxilong/sentiment-analysis-on-movie-reviews/cfg/flan-t5-base.yaml -o "per_device_train_batch_size=256|per_device_eval_batch_size=256|num_train_epochs=3|learning_rate=2e-4"

twitter-roberta-base-sentiment-latest
CUDA_VISIBLE_DEVICES=4 python /risk1/chengxilong/sentiment-analysis-on-movie-reviews/train.py -l /risk1/chengxilong/sentiment-analysis-on-movie-reviews/runs/twitter-roberta-base-sentiment-latest-0 -mc /risk1/chengxilong/sentiment-analysis-on-movie-reviews/cfg/twitter-roberta-base-sentiment-latest.yaml -o "per_device_train_batch_size=256|per_device_eval_batch_size=256|num_train_epochs=3|learning_rate=2e-4"

xlnet-base-cased
CUDA_VISIBLE_DEVICES=4 python /risk1/chengxilong/sentiment-analysis-on-movie-reviews/train.py -l /risk1/chengxilong/sentiment-analysis-on-movie-reviews/runs/xlnet-base-cased-0 -mc /risk1/chengxilong/sentiment-analysis-on-movie-reviews/cfg/xlnet-base-cased.yaml -o "per_device_train_batch_size=256|per_device_eval_batch_size=256|num_train_epochs=3|learning_rate=2e-4"