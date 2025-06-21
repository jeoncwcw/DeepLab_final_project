import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class NormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormalizedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x_norm, w_norm)

class CifarResNet18(nn.Module):
    def __init__(self, num_classes=100, use_norm=False):
        super(CifarResNet18, self).__init__()
               
        backbone = resnet18(pretrained=False, num_classes=num_classes)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.bn1 = nn.BatchNorm2d(64)
        backbone.maxpool = nn.Identity()

        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        in_features = backbone.fc.in_features

        if use_norm:    # for LDAM
            self.classifier = NormalizedLinear(in_features, num_classes)
        else:
            self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        features = self.encoder(x).flatten(1)
        logits = self.classifier(features)         
        return logits
    
class CifarResNet18_ThreeStage(nn.Module):
    def __init__(self, num_classes=100, feat_dim=128, cls_num_list = None):
        super(CifarResNet18_ThreeStage, self).__init__()
        backbone = resnet18(pretrained=False, num_classes=num_classes)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.bn1 = nn.BatchNorm2d(64)
        backbone.maxpool = nn.Identity()

        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        in_features = backbone.fc.in_features
        self.cls_num_list = cls_num_list

        self.projection_head = nn.Sequential(
            nn.Linear(in_features, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

        self.ldam_classifier = NormalizedLinear(in_features, num_classes)
        self.cse_classifier = nn.Linear(in_features, num_classes)

        self.training_stage = 1
    
    def set_training_stage(self, stage):
        self.training_stage = stage
        assert stage in [1, 2, 3]
        match stage:
            case 1:
                for param in self.encoder.parameters():
                    param.requires_grad = True
                for param in self.projection_head.parameters():
                    param.requires_grad = True
                for param in self.ldam_classifier.parameters():
                    param.requires_grad = False
                for param in self.cse_classifier.parameters():
                    param.requires_grad = False
            case 2:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                for param in self.projection_head.parameters():
                    param.requires_grad = False
                for param in self.ldam_classifier.parameters():
                    param.requires_grad = True
                for param in self.cse_classifier.parameters():
                    param.requires_grad = False
            case 3:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                for param in self.projection_head.parameters():
                    param.requires_grad = False
                for param in self.ldam_classifier.parameters():
                    param.requires_grad = False
                for param in self.cse_classifier.parameters():
                    param.requires_grad = True
                
    def forward(self, x, inference_mode=False):
        features = self.encoder(x).flatten(1)
        if inference_mode:
            return self._confidence_based_prediction(features)
            # return self._fixed_softgate_logits(features)

        assert self.training_stage in [1, 2, 3]
        match self.training_stage:
            case 1:
                scl_features = F.normalize(self.projection_head(features), dim=1)
                return scl_features
            case 2:
                ldam_logits = self.ldam_classifier(features)
                return ldam_logits
            case 3:
                cse_logits = self.cse_classifier(features)
                return cse_logits
        
    def _confidence_based_prediction(self, features):
        with torch.no_grad():
            ldam_logits = self.ldam_classifier(features) * 30
            cse_logits = self.cse_classifier(features)

            ldam_probs = F.softmax(ldam_logits, dim=1)
            cse_probs = F.softmax(cse_logits , dim=1)

            ldam_conf = ldam_probs.max(dim=1)[0]
            cse_conf = cse_probs.max(dim=1)[0]

            use_ldam = (ldam_conf > cse_conf).unsqueeze(1).float()
            final_logits = use_ldam * ldam_logits + (1 - use_ldam) * cse_logits
        return cse_logits
    
    def get_individual_predictions(self, x):
        features = self.encoder(x).flatten(1)

        with torch.no_grad():
            ldam_logits = self.ldam_classifier(features)
            cse_logits = self.cse_classifier(features)

            ldam_probs = F.softmax(ldam_logits, dim=1)
            cse_probs = F.softmax(cse_logits, dim=1)

            ldam_conf = ldam_probs.max(dim=1)[0]
            cse_conf = cse_probs.max(dim=1)[0]

        return {
            'ldam_logits': ldam_logits,
            'cse_logits': cse_logits,
            'ldam_conf': ldam_conf,
            'cse_conf': cse_conf,
            'ldam_preds': ldam_logits.argmax(dim=1),
            'cse_preds': cse_logits.argmax(dim=1)
        }
    
    def _fixed_softgate_logits(self, features):
        freq = torch.tensor(self.cls_num_list, dtype=torch.float32)
        n_head = freq.max()
        alpha = (n_head.sqrt() / (freq.sqrt() + n_head.sqrt())).to(features.device)

        ldam_logits = self.ldam_classifier(features) * 30
        cse_logits = self.cse_classifier(features)

        ldam_logits_norm = (ldam_logits - ldam_logits.mean(dim=1, keepdim=True)) / ldam_logits.std(dim=1, keepdim=True)
        cse_logits_norm = (cse_logits - cse_logits.mean(dim=1, keepdim=True)) / cse_logits.std(dim=1, keepdim=True)

        alpha = alpha.unsqueeze(0)
        final_logits = alpha * ldam_logits_norm + (1 - alpha) * cse_logits_norm

        return final_logits