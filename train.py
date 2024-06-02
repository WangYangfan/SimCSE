import os
import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import BertTokenizer
import datasets
from datasets import DatasetDict
from accelerate import Accelerator
import evaluate

from loguru import logger
from tqdm.auto import tqdm
import argparse
from dotenv import load_dotenv

from utils import seed_environment
from model.simcse import SimCSE
from loss.simcseloss import YangJXSimCSEUnSupLoss, SimCSEUnSupLoss, SimCSESupLoss, YangJXSimCSESupLoss



def preprocess_wiki(examples) -> dict:
    inputs = {}
    sentence_list = [s.strip() for s in examples['sentence']]
    sentence_tokenized = []
    for s in sentence_list:
        s_td = tokenizer(
            [s, s],
            max_length=args.max_seq_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        sentence_tokenized.append(s_td)
    inputs['sentence'] = sentence_tokenized
    return inputs

def preprocess_nli(examples) -> dict:
    inputs = {}
    sent1_list = [s1.strip() for s1 in examples['sent1']]
    sent2_list = [s2.strip() for s2 in examples['sent2']]
    neg_list = [e.strip() for e in examples['hard_neg']]
    sentence_tokenized = []
    for s1, s2, e in zip(sent1_list, sent2_list, neg_list):
        s_td = tokenizer(
            [s1, s2, e],
            max_length=args.max_seq_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        sentence_tokenized.append(s_td)
    inputs['sentence'] = sentence_tokenized
    return inputs

def preprocess_stsbenchmark(examples) -> dict:
    inputs = {}
    sent1_list = [s1.strip() for s1 in examples['sent1']]
    sent2_list = [s2.strip() for s2 in examples['sent2']]
    scores = [float(sc) for sc in examples['score']]

    sent1_tokenized = []
    sent2_tokenized = []
    for s1, s2 in zip(sent1_list, sent2_list):
        s1_td = tokenizer(
            s1,
            max_length=args.max_seq_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        s2_td = tokenizer(
            s2,
            max_length=args.max_seq_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        sent1_tokenized.append(s1_td)
        sent2_tokenized.append(s2_td)

    inputs['sent1'] = sent1_tokenized
    inputs['sent2'] = sent2_tokenized
    inputs['score'] = scores
    return inputs

def collator_wiki_nli(batch) -> dict:
    sentences = {}
    sentences['input_ids'] = torch.cat([x['sentence']['input_ids'] for x in batch], dim=0)   # [batch_size*2, max_seq_len]
    sentences['attention_mask'] = torch.cat([x['sentence']['attention_mask'] for x in batch], dim=0)
    sentences['token_type_ids'] = torch.cat([x['sentence']['token_type_ids'] for x in batch], dim=0)
    return sentences

def collator_stsbenchmark(batch) -> dict:
    sent1s, sent2s, scores = {}, {}, {}
    sent1s['input_ids'] = torch.cat([x['sent1']['input_ids'] for x in batch], dim=0)
    sent1s['attention_mask'] = torch.cat([x['sent1']['attention_mask'] for x in batch], dim=0)
    sent1s['token_type_ids'] = torch.cat([x['sent1']['token_type_ids'] for x in batch], dim=0)

    sent2s['input_ids'] = torch.cat([x['sent2']['input_ids'] for x in batch], dim=0)
    sent2s['attention_mask'] = torch.cat([x['sent2']['attention_mask'] for x in batch], dim=0)
    sent2s['token_type_ids'] = torch.cat([x['sent2']['token_type_ids'] for x in batch], dim=0)

    scores['score'] = torch.cat([x['score'].unsqueeze(0) for x in batch], dim=0)
    return sent1s, sent2s, scores

def evaluator(model: nn.Module, dataloader: DataLoader):
    model.eval()
    for batch1, batch2, label in tqdm(dataloader, desc="Evaluaing[STSBenchmark]", unit="batch"):
        pool_out1, _ = model(**batch1)
        pool_out2, _ = model(**batch2)
        prediction = F.cosine_similarity(pool_out1, pool_out2, dim=-1)  # [batch_size]

    prediction_gathered = accelerator.gather(prediction)
    label_gathered = accelerator.gather(label['score'])

    spearmanr_metric.add_batch(predictions=prediction_gathered, references=label_gathered)
    spearmanr_score = spearmanr_metric.compute()['spearmanr']
    return spearmanr_score

if __name__ == '__main__':
    load_dotenv(dotenv_path="./envs/simcse.env", verbose=True, override=True)
    checkpoint = os.getenv('CHECKPOINT')
    logger.info("🐳 checkpoint: {}".format(checkpoint))

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default="../princeton_data/wiki")
    parser.add_argument('--eval_dataset', type=str, default="../princeton_data/sts")
    parser.add_argument('--output_path', type=str, default="./output/")
    parser.add_argument('--save_info', type=str, default="")
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--max_seq_length', type=int, default=32)
    parser.add_argument('--pooling', type=str, default='cls')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--unfrozen_layers', type=str, nargs='+', default=['all_layers'])
    parser.add_argument('--num_shards', type=int, default=None)
    args = parser.parse_args()
    assert args.save_info in ['unsup', 'yangjx_unsup', 'sup', 'yangjx_sup']
    assert args.pooling in ['cls', 'pooler', 'last-avg', 'last2-avg', 'first-last-avg']
    logger.info("🍻 args: {}".format(args))

    seed_environment(seed=args.seed)
    accelerator = Accelerator()
    device = accelerator.device
    logger.info("🧊 device: {}".format(device))

    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    model = SimCSE(checkpoint=checkpoint, pooling=args.pooling, dropout=args.dropout, unfrozen_layers=args.unfrozen_layers)
    logger.info("🥶 unfrozen layers: {}".format(args.unfrozen_layers))
    # check_model(model)

    train_dataset = datasets.load_from_disk(args.train_dataset).shuffle(seed=args.seed)
    # print(dataset)
    if args.num_shards is not None:
        new_dataset = DatasetDict()
        new_dataset['train'] = train_dataset['train'].shard(num_shards=args.num_shards, index=0)
        train_dataset = new_dataset
    # print(dataset['train'][0])
    if args.save_info in ['unsup', 'yangjx_unsup']:
        train_dataset_tokenized = train_dataset.map(preprocess_wiki, batched=True)
    elif args.save_info in ['sup', 'yangjx_sup']:
        train_dataset_tokenized = train_dataset.map(preprocess_nli, batched=True)
    else:
        raise NotImplementedError
    train_dataset_tokenized.set_format('pt')

    eval_dataset = datasets.load_from_disk(args.eval_dataset)
    eval_dataset_tokenized = eval_dataset.map(preprocess_stsbenchmark, batched=True)
    eval_dataset_tokenized.set_format('pt')
    logger.info("💾 train: {}, valid: {}, test: {}".format(len(train_dataset['train']), len(eval_dataset['validation']), len(eval_dataset['test'])))
  
    train_dataloader = DataLoader(
        dataset=train_dataset_tokenized['train'],
        shuffle=False,
        collate_fn=collator_wiki_nli,
        batch_size=args.batch_size,
    )
    valid_dataloader = DataLoader(
        dataset=eval_dataset_tokenized['validation'],
        shuffle=False,
        collate_fn=collator_stsbenchmark,
        batch_size=args.batch_size * 2,
    )
    test_dataloader = DataLoader(
        dataset=eval_dataset_tokenized['test'],
        shuffle=False,
        collate_fn=collator_stsbenchmark,
        batch_size=args.batch_size,
    )

    spearmanr_metric = evaluate.load("../metric/spearmanr.py")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    if args.save_info == 'yangjx_unsup':
        criterion = YangJXSimCSEUnSupLoss(device=device, temperature=0.05)
    elif args.save_info == 'unsup':
        criterion = SimCSEUnSupLoss(device=device, temperature=0.05)
    elif args.save_info == 'sup':
        criterion = SimCSESupLoss(device=device, temperature=0.05)
    elif args.save_info == 'yangjx_sup':
        criterion = YangJXSimCSESupLoss(device=device, temperature=0.05)
    else:
        raise NotImplementedError

    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )
    logger.info("👏 load model, dataset successfully!")

    # 计算len(dataloader)一定要在prepare之后
    num_train_epochs = args.num_epochs
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    progress_bar = tqdm(range(num_training_steps), desc="Training[WIKI]", unit="step")
    global_step = 0
    for epoch in range(num_train_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            _, mlp_out = model(**batch)
            loss = criterion(mlp_out)
            accelerator.backward(loss)
            # check_model(model)
            epoch_loss += loss.item()
            # if global_step % 10 == 0:
            logger.info(">>> (train) loss: {}".format(loss.item()))
            
            # print(loss.item(), loss.grad, loss.grad_fn, loss.requires_grad)
            # print(mlp_out.grad, mlp_out.grad_fn, mlp_out.requires_grad)

            optimizer.step()
            progress_bar.update(1)
            global_step += 1
        epoch_loss /= num_update_steps_per_epoch

        spearmanr_score = evaluator(model=model, dataloader=valid_dataloader)
        logger.info(">>> (train) epoch loss: {}, (valid) spearmanr score: {}".format(epoch_loss, spearmanr_score))

        if accelerator.is_main_process:
            accelerator.wait_for_everyone()
            logger.info("🌏 Everyone is here!")

            save_path = os.path.join(
                args.output_path, 
                "{}".format(args.save_info), 
                "checkpoint-{}".format(global_step)
            )
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            accelerator.save_state(save_path)
            logger.info("Saved state to {}".format(save_path), main_process_only=True)

    spearmanr_score = evaluator(model=model, dataloader=test_dataloader)
    logger.info(">>> (test) spearmanr score: {}".format(spearmanr_score))
