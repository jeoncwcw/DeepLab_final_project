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

class DRWLDAMLoss(nn.Module):
    """DRW가 적용된 LDAM Loss - 개선된 버전"""
    def __init__(self, cls_num_list, max_m=0.5, s=30, drw_start_ratio=0.75):
        super(DRWLDAMLoss, self).__init__()
        self.cls_num_list = cls_num_list
        self.max_m = max_m
        self.s = s
        self.drw_start_ratio = drw_start_ratio
        self.current_epoch = 0
        self.total_epochs = 60  # 기본값, update_epoch에서 갱신됨
        
        # 기본 LDAM 설정 (초기)
        self.base_ldam = LDAMLoss(cls_num_list, max_m, s)
        
        # DRW용 클래스 가중치 미리 계산
        weights = 1.0 / np.array(cls_num_list)
        weights = weights / np.sum(weights) * len(cls_num_list)
        self.class_weights = torch.FloatTensor(weights)
        if torch.cuda.is_available():
            self.class_weights = self.class_weights.cuda()
        
        # Enhanced LDAM (DRW 적용 시 사용) - 미리 생성해서 재사용
        self.enhanced_ldam = None
        self.drw_active = False
    
    def update_epoch(self, epoch, total_epochs):
        """에폭 정보 업데이트 및 DRW 상태 확인"""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        
        drw_start_epoch = int(self.total_epochs * self.drw_start_ratio)
        
        # DRW 상태가 변경되었을 때만 enhanced_ldam 생성
        if epoch >= drw_start_epoch and not self.drw_active:
            self.drw_active = True
            # Enhanced LDAM with class weights 생성 (한 번만)
            self.enhanced_ldam = self._create_weighted_ldam()
            print(f"[DRW Activated] Epoch {epoch+1}: Enhanced LDAM with class reweighting applied")
    
    def _create_weighted_ldam(self):
        """가중치가 적용된 LDAM 생성"""
        class WeightedLDAMLoss(nn.Module):
            def __init__(self, cls_num_list, max_m, s, class_weights):
                super().__init__()
                # Enhanced margin 설정
                m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
                m_list = m_list * (max_m / np.max(m_list))
                m_list = torch.FloatTensor(m_list)
                if torch.cuda.is_available():
                    m_list = m_list.cuda()
                self.m_list = m_list
                self.s = s
                self.class_weights = class_weights
            
            def forward(self, x, target):
                # LDAM margin 적용
                batch_margins = self.m_list[target]
                index_mask = torch.zeros_like(x, dtype=torch.bool)
                index_mask.scatter_(1, target.data.view(-1, 1), 1)
                
                x_m = x.clone()
                x_m[index_mask] -= batch_margins
                output = self.s * x_m
                
                # 가중치가 적용된 CrossEntropyLoss 사용
                return nn.CrossEntropyLoss(weight=self.class_weights)(output, target)
        
        return WeightedLDAMLoss(
            cls_num_list=self.cls_num_list, 
            max_m=0.8,  # Enhanced margin
            s=40,       # Enhanced scaling
            class_weights=self.class_weights
        )
    
    def forward(self, x, target):
        if self.drw_active and self.enhanced_ldam is not None:
            # DRW 적용: Enhanced LDAM with class weights
            return self.enhanced_ldam(x, target)
        else:
            # 일반 LDAM (초기 단계)
            return self.base_ldam(x, target)


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
    elif mode.lower() == "drw_ldam":  # 새로운 모드 추가
        if cls_num_list is None:
            raise ValueError("[DRW LDAM Loss를 위해선 cls_num_list 필요]")
        return DRWLDAMLoss(cls_num_list=cls_num_list)
    elif mode.lower() == "scl":
        if cls_num_list:
            return SupConLoss(temperature=0.05, cls_num_list=cls_num_list)
        return SupConLoss(temperature=0.05)
