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
    def __init__(self, temperature=0.2,  class_weights=None, balance_alpha=0.5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.class_weights = class_weights
        self.balance_alpha = balance_alpha

    def forward(self, features, labels=None):
        if len(features.shape) != 3: raise ValueError("3차원 이어야 함")

        device = features.device
        batch_size = features.shape[0]
        num_views = features.shape[1]

        labels = labels.repeat(num_views).view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)

        self_contrast_mask = torch.eye(batch_size * num_views, device=device)
        logits_mask = 1 - self_contrast_mask
        mask = mask * logits_mask

        if self.class_weights is not None:
            class_weights = self.class_weights[labels.squeeze()]
            class_weights = class_weights.view(-1, 1)
            mask = mask * class_weights

        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T),
            self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask

        neg_mask = 1 - mask
        neg_weights = torch.ones_like(neg_mask) * self.balance_alpha

        combined_weights = mask + neg_weights * neg_mask
        log_prob = logits - torch.log((exp_logits * combined_weights).sum(1, keepdim=True) + 1e-8)

        mask_sum = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1)[mask_sum>0] / (mask_sum[mask_sum > 0] + 1e-8)

        loss = -mean_log_prob_pos

        if loss.numel() == 0:
            return torch.tensor(0.0).to(device)
        
        loss = loss.mean()
        return loss





        



def myLoss(mode = "CrossEntropy", cls_num_list=None):
    if mode.lower() == "crossentropy":
        return nn.CrossEntropyLoss()
    elif mode.lower() == "ldam":
        if cls_num_list is None:
            raise ValueError("[LDAM Loss를 위해선 cls_num_list 필요]")
        return LDAMLoss(cls_num_list=cls_num_list)
    elif mode == "SCL":
        if cls_num_list is not None:
            cls_num_list = np.array(cls_num_list)
            class_weights = 1.0 / np.sqrt(cls_num_list)
            class_weights = class_weights / np.max(class_weights)
            class_weights = torch.FloatTensor(class_weights)
            if torch.cuda.is_available():
                class_weights = class_weights.cuda()
        else:
            class_weights = None
        
        return SupConLoss(class_weights=class_weights)
