import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30):
        """
        LDAM Loss 초기화 함수
        Args:
            cls_num_list (list): 클래스별 샘플 개수를 담은 리스트
            max_m (float): 최대 마진 값 (하이퍼파라미터 C). 클래스별 마진은 이 값을 기준으로 계산
            s (int): Scaling factor
            weight (tensor, optional): 클래스별 가중치. Defaults to None.
        """
        super(LDAMLoss, self).__init__()
        
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        if torch.cuda.is_available():
            m_list = m_list.cuda()
        self.m_list = m_list

        assert s > 0
        self.s = s

    def forward(self, x, target):
        batch_margins = self.m_list[target]
        index_mask = torch.zeros_like(x, dtype=torch.bool)
        index_mask.scatter_(1, target.data.view(-1, 1), 1)

        x_m = x.clone()
        x_m[index_mask] -= batch_margins

        output = self.s * x_m
        return nn.CrossEntropyLoss()(output, target)



class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    참고: Khosla et al., 2020 (https://arxiv.org/abs/2004.11362)
    """
    def __init__(self, temperature: float = 0.07, cls_num_list = None, weight_power=0.5):
        super().__init__()
        self.temperature = temperature
        self.weight_power = weight_power

        if cls_num_list is not None:
            cls_nums = torch.tensor(cls_num_list, dtype=torch.float32)
            max_n = cls_nums.max()
            self.cls_weights = (max_n / cls_nums) ** weight_power
        else:
            self.cls_weights = None

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Args
        ----
        features : (B, V, C)  # V = views per sample
        labels   : (B,)       # int64
        """
        if features.ndim != 3:
            raise ValueError("features shape must be (B, V, C)")
        device   = features.device        

  
        B, V, C  = features.shape

        # (B*V, C)
        features = F.normalize(features.reshape(B * V, C), dim=1)

        # label mask
        labels = labels.view(B, 1).repeat(1, V).reshape(-1)
        mask   = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(device)

        # similarity logits
        logits  = torch.div(torch.matmul(features, features.T), self.temperature)

        # self-contrast 제거
        logits_mask = torch.ones_like(mask) - torch.eye(B * V, device=device)
        mask        = mask * logits_mask

        # stability trick
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask            # negative weight = 1
        log_prob   = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1e-8)

        if self.cls_weights is not None:
            sample_weights = self.cls_weights.to(device)[labels]
            loss = -(sample_weights * mean_log_prob_pos).sum() / sample_weights.sum()
        else:
            loss = -mean_log_prob_pos.mean()

        return loss  * self.temperature / 0.07





        



def myLoss(mode = "CrossEntropy", cls_num_list=None):
    if mode.lower() == "crossentropy":
        return nn.CrossEntropyLoss()
    elif mode.lower() == "ldam":
        if cls_num_list is None:
            raise ValueError("[LDAM Loss를 위해선 cls_num_list 필요]")
        return LDAMLoss(cls_num_list=cls_num_list)
    elif mode.lower() == "scl":
        if cls_num_list:
            return SupConLoss(temperature=0.05, cls_num_list=cls_num_list)
        return SupConLoss(temperature=0.05)
