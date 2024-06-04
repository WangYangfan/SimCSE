import torch
import torch.nn.functional as F


class SimCSEUnSupLoss:
    """
        SimCSE unsup loss from: https://github.com/princeton-nlp/SimCSE
    """
    def __init__(self, device, temperature=0.05):
        self.temperature = temperature
        self.device = device

    def __call__(self, features: torch.Tensor):
        batch_size = features.shape[0] // 2

        z = features.view(batch_size, 2, -1)
        z1, z2 = z[:, 0], z[:, 1]
        sim_mat = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1) / self.temperature

        labels = torch.arange(sim_mat.shape[0], device=self.device).long()

        loss = F.cross_entropy(sim_mat, labels)

        return loss
    
class SimCSESupLoss:
    """ 
        SimCSE sup loss from: https://github.com/princeton-nlp/SimCSE
        仅针对原论文提到的nli数据，每个样本是（原始，正，负）组成的三元组 
    """
    def __init__(self, device, temperature=0.05):
        self.temperature = temperature
        self.device = device

    def __call__(self,  features: torch.Tensor):
        batch_size = features.shape[0] // 3

        z = features.view(batch_size, 3, -1)
        z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2]
        z1_z2_cos = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1) / self.temperature
        z1_z3_cos = F.cosine_similarity(z1.unsqueeze(1), z3.unsqueeze(0), dim=-1) / self.temperature
        sim_mat = torch.cat([z1_z2_cos, z1_z3_cos], dim=1)

        labels = torch.arange(sim_mat.shape[0], device=self.device).long()

        # contradiction as hard negatives in paper for nli
        # 实际上是在负样本处硬加上一个常量z3_weight
        z3_weight = 0
        weights = torch.tensor(
            [[0.0] * (sim_mat.shape[-1] - z1_z3_cos.shape[-1]) + [0.0] * i + [z3_weight] + \
             [0.0] * (z1_z3_cos.shape[-1] - i - 1) for i in range(z1_z3_cos.shape[-1])],
            device=self.device
        )
        sim_mat = sim_mat - weights

        loss = F.cross_entropy(sim_mat, labels)
        return loss
    

class YangJXSimCSEUnSupLoss:
    """ 
        SimCSE unsup loss from: https://github.com/yangjianxin1/SimCSE 
        对原论文计算相似度的样本组合方式做了改进
    """
    def __init__(self, device, temperature=0.05):
        self.temperature = temperature
        self.device = device

    def __call__(self, features: torch.Tensor):
        labels = torch.arange(features.shape[0]).to(self.device)
        labels = (labels - labels % 2 * 2) + 1
        # 定义每个样本对应的正样本的序号

        sim_mat = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        sim_mat = sim_mat - torch.eye(sim_mat.shape[0], device=self.device) * 1e12
        sim_mat = sim_mat / self.temperature
        # 计算句向量余弦相似度矩阵，对角线置为负无穷，使用温度系数放大

        loss = F.cross_entropy(sim_mat, labels)
        return loss
    
class YangJXSimCSESupLoss:
    """ 
        SimCSE unsup loss from: https://github.com/yangjianxin1/SimCSE 
        和原论文计算相似度的样本组合方式基本一致
    """
    def __init__(self, device, temperature=0.05):
        self.temperature = temperature
        self.device = device

    def __call__(self, features: torch.Tensor):
        sim_mat = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        raw = torch.arange(0, features.shape[0], 3)
        raw_pos_neg = torch.arange(0, features.shape[0])
        pos_neg = raw_pos_neg[raw_pos_neg % 3 != 0]
        sim_mat = sim_mat[raw, :]
        sim_mat = sim_mat[:, pos_neg]
        sim_mat = sim_mat / self.temperature

        labels = torch.arange(0, len(pos_neg), 2, device=self.device)

        loss = F.cross_entropy(sim_mat, labels)
        return loss
