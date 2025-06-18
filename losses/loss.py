import torch.nn as nn
import torch
import numpy as np

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








def myLoss(mode = "CrossEntropy", cls_num_list=None):
    if mode == "CrossEntropy":
        return nn.CrossEntropyLoss()
    elif mode == "LDAM":
        if cls_num_list is None:
            raise ValueError("[LDAM Loss를 위해선 cls_num_list 필요]")
        return LDAMLoss(cls_num_list=cls_num_list)