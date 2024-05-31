# SimCSE

🐣 a simple implementation of SimCSE.

**References**
- original paper： [Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821)
- official repositories: [princeton-nlp/SimCSE](https://github.com/princeton-nlp/SimCSE)

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
