# SimCSE

🐣 a simple implementation of SimCSE.

**References**
- original paper： [Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821)
- official repositories: [princeton-nlp/SimCSE](https://github.com/princeton-nlp/SimCSE)
- good implementation: [yangjianxin1/SimCSE](https://github.com/yangjianxin1/SimCSE)

**Requirements**

```
python==3.9.19
torch==2.3.0
transformers==4.40.2
accelerate==0.30.1
datasets==2.19.1
evaluate==0.4.2
tensorboard
argparse
loguru
python-dotenv
```

## Data Preparation

**Download Datasets for Training**

dataset for supervised training:

```bash
wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/nli_for_simcse.csv
```

dataset for unsupervised training:

```bash
wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt
```

**Download SentEval**

dataset for evaluation:

```bash
wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/senteval.tar
tar xvf senteval.tar
```

**🤣 Download Datasets: One for all**

```bash
huggingface-cli download --repo-type dataset --resume-download princeton-nlp/datasets-for-simcse --local-dir data
```

**load datasets**

```python
python data/load_datasets.py
```

## Train and Eval

**Ensure**

```python
assert args.save_info in ['unsup', 'yangjx_unsup', 'sup', 'yangjx_sup']
assert args.pooling in ['cls', 'pooler', 'last-avg', 'last2-avg', 'first-last-avg']
```

**Train**

For unsup setting, we use `wiki1m` dataset. Also, you can select loss function from [princeton-nlp/SimCSE](https://github.com/princeton-nlp/SimCSE) or [yangjianxin1/SimCSE](https://github.com/yangjianxin1/SimCSE) by changing `save_info`.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --train_dataset ../princeton_data/wiki \
    --eval_dataset ../princeton_data/sts \
    --output_path ./output \
    --save_info unsup \
    --batch_size 64 \
    --lr 3e-5 \
    --num_epochs 5 \
    --max_seq_length 32 \
    --pooling cls \
    --dropout 0.1 \
    --seed 42
```

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
    --train_dataset ../princeton_data/wiki \
    --eval_dataset ../princeton_data/sts \
    --output_path ./output \
    --save_info yangjx_unsup \
    --batch_size 64 \
    --lr 3e-5 \
    --num_epochs 5 \
    --max_seq_length 32 \
    --pooling cls \
    --dropout 0.1 \
    --seed 42
```

For sup setting, we use `nli` dataset. Also, you can select loss function from [princeton-nlp/SimCSE](https://github.com/princeton-nlp/SimCSE) or [yangjianxin1/SimCSE](https://github.com/yangjianxin1/SimCSE) by changing `save_info`.

```bash
CUDA_VISIBLE_DEVICES=2 python train.py \
    --train_dataset ../princeton_data/nli \
    --eval_dataset ../princeton_data/sts \
    --output_path ./output \
    --save_info sup \
    --batch_size 64 \
    --lr 3e-5 \
    --num_epochs 5 \
    --max_seq_length 32 \
    --pooling cls \
    --dropout 0.1 \
    --seed 42
```

```bash
CUDA_VISIBLE_DEVICES=3 python train.py \
    --train_dataset ../princeton_data/nli \
    --eval_dataset ../princeton_data/sts \
    --output_path ./output \
    --save_info yangjx_sup \
    --batch_size 64 \
    --lr 3e-5 \
    --num_epochs 5 \
    --max_seq_length 32 \
    --pooling cls \
    --dropout 0.1 \
    --seed 42
```

Arguments:
- `--train_dataset`: train dataset path;
- `--eval_dataset`: eval dataset path;
- `--output_path`: parent path to save trained model;
- `--save_info`: select loss function for train, also `ouput_path/save_info/checkpoint-{}` to save trained model;
- `--num_shards`: split train dataset to `num_shards` shards;
- `--unfrozen_layers`: default `['all_layers']`. if set, only layers in `unfrozen_layers` are trainable;
- `--batch_size`: `batch_size` for train, `batch_size` for test, 2 * `batch_size` for validation;
- `--lr`: learning rate;
- `--num_epochs`: number of train epochs;
- `--max_seq_length`: max seq length for tokenizer;
- `--pooling`: select pooling mode for bert;
- `--dropout`: dropout argument for bert;
- `--seed`: random seed for environment.

**Eval**

For evaluation, we only select `SentEval/STS/STSBenchmark/sts-dev.csv` and `SentEval/STS/STSBenchmark/sts-test.csv` during valid and test.


