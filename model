import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

import sys
sys.path.append("..")
from utils import frozen_bert, check_model

class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features):
        x = self.dense(features)
        out = self.activation(x)

        return out
    

class Pooler(nn.Module):
    def __init__(self, pooling):
        super(Pooler, self).__init__()
        self.pooling = pooling
        assert self.pooling in ['cls', 'pooler', 'last-avg', 'last2-avg', 'first-last-avg']

    def forward(self, attention_mask: torch.Tensor, outputs):
        last_hidden: torch.Tensor = outputs.last_hidden_state # [batch_size, max_seq_length, feature_size]
        pooler_output: torch.Tensor = outputs.pooler_output   # [batch_size, feature_size]
        hidden_states: torch.Tensor = outputs.hidden_states   # [num_layers, batch_size, max_seq_length, feature_size]

        if self.pooling == 'cls':
            return last_hidden[:, 0, :] 
        if self.pooling == 'pooler':
            return pooler_output
        if self.pooling == 'last-avg':
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        if self.pooling == 'last2-avg':
            second_last_hidden = hidden_states[-2]
            return (((last_hidden + second_last_hidden) / 2.0) * attention_mask.unsqueeze(-1)).sum(1) / \
                attention_mask.sum(-1).unsqueeze(-1)
        if self.pooling == 'first-last-avg':
            first_hidden = hidden_states[1] #WHY
            return (((last_hidden + first_hidden) / 2.0) * attention_mask.unsqueeze(-1)).sum(1) / \
                attention_mask.sum(-1).unsqueeze(-1)
        
        raise NotImplementedError
        

class SimCSE(nn.Module):
    """ SimCSE: 原论文中写到pool='cls'且仅在训练时使用mlp的设置效果最好 """
    # the token-level MLM objective improves the averaged performance on transfer tasks modestly, 
    # yet it brings a consistent drop in semantic textual similarity tasks.

    # For both unsupervised and supervised SimCSE, we take the [CLS] representation with an MLP
    # layer on top of it as the sentence representation. Specially, for unsupervised SimCSE, we discard
    # the MLP layer and only use the [CLS] output during test, since we find that it leads to better performance (ablation study in §6.3).
    
    # 如果DDP报错，使用TORCH_DISTRIBUTED_DEBUG=DETAIL查询未参与计算的参数
    def __init__(self, checkpoint, pooling='cls', dropout=0.1, unfrozen_layers=[]):
        super(SimCSE, self).__init__()
        bert_config = BertConfig.from_pretrained(checkpoint)
        bert_config.attention_probs_dropout_prob = dropout
        bert_config.hidden_dropout_prob = dropout

        self.bert = BertModel.from_pretrained(checkpoint, config=bert_config)
        self.pooler = Pooler(pooling)
        self.mlp = MLP(bert_config.hidden_size)

        frozen_bert(self.bert, unfrozen_layers)
        # check_model(self.bert)

    def forward(self, input_ids, attention_mask, token_type_ids):

        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        pool_out = self.pooler(attention_mask, out)
        mlp_out = self.mlp(pool_out)
        
        return pool_out, mlp_out
