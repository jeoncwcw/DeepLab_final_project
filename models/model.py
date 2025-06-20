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
    def __init__(self, num_classes=100, feat_dim=128):
        super(CifarResNet18_ThreeStage, self).__init__()
        backbone = resnet18(pretrained=False, num_classes=num_classes)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.bn1 = nn.BatchNorm2d(64)
        backbone.maxpool = nn.Identity()

        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        in_features = backbone.fc.in_features

        self.projection_head = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, feat_dim)
        )

        self.ldam_classifier = NormalizedLinear(in_features, num_classes)
        self.cse_classifier = nn.Linear(in_features, num_classes)

        self.training_stage = 1
    
    def set_training_stage(self, stage):
        self.training_stage = stage
        if stage==3:
            for param in self.ldam_classifier.parameters():
                param.requires_grad = False
            for param in self.projection_head.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, x, inference_mode=False):
        features = self.encoder(x).flatten(1)
        if inference_mode:
            return self._confidence_based_prediction(features)

        assert self.training_stage in [1, 2, 3]
        if self.training_stage == 1:
            logits = self.ldam_classifier(features)
            return logits
        elif self.training_stage == 2:
            ldam_logits = self.ldam_classifier(features)
            scl_features = F.normalize(features, dim=1)
            return scl_features, ldam_logits
        elif self.training_stage == 3:
            cse_logits = self.cse_classifier(features)
            return cse_logits
        
    def _confidence_based_prediction(self, features):
        with torch.no_grad():
            ldam_logits = self.ldam_classifier(features)
            cse_logits = self.cse_classifier(features)

            ldam_probs = F.softmax(ldam_logits, dim=1)
            cse_probs = F.softmax(cse_logits, dim=1)

            ldam_conf = ldam_probs.max(dim=1)[0]
            cse_conf = cse_probs.max(dim=1)[0]

            use_ldam = (ldam_conf > cse_conf).unsqueeze(1).float()
            final_logits = use_ldam * ldam_logits + (1 - use_ldam) * cse_logits
        return final_logits
    
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