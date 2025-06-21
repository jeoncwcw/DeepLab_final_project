import math
import torch
from torch.optim.lr_scheduler import LambdaLR
from losses.loss import myLoss

def create_stage_scheduler(stage, epochs):
    warmup_epochs = 5
    match stage:
        case 1:
            def stage1_scheduler(epoch):
                if epoch < warmup_epochs:
                    return float(epoch+1) / warmup_epochs
                else:
                    progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
            return stage1_scheduler
        case 2:
            def stage2_scheduler(epoch):
                if epoch < warmup_epochs:
                    return float(epoch+1) / warmup_epochs
                else:
                    progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
                    return 0.5 * (1.0 + math.cos(math.pi * progress * 0.7))
            return stage2_scheduler
        case 3:
            def stage3_scheduler(epoch):
                return 1 - (epoch/epochs)
            return stage3_scheduler
    
def get_stage_configs():
    return {
        "stage1": {
            "epochs": 100,
            "lr": 0.2,
            "model_save_path": "./models_path/stage1_SCL_only.pth"
        },
        "stage2": { 
            "epochs": 60,
            "lr": 0.02,
            "model_save_path": "./models_path/stage2_LDAM.pth"
        },
        "stage3": {
            "epochs": 60, 
            "lr": 0.05,
            "model_save_path": "./models_path/stage3_CSE.pth"
        }
    }


def create_stage_config(stage, cls_num_list, model):
    stage_configs = get_stage_configs()
    config = stage_configs[f"stage{stage}"]

    match stage:
        case 1:
            criterion = myLoss(mode="SCL", cls_num_list=cls_num_list)
        case 2:
            criterion = myLoss(mode="LDAM", cls_num_list=cls_num_list)
        case 3:
            criterion = myLoss(mode="CrossEntropy")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=5e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=create_stage_scheduler(stage, config["epochs"]))

    full_config = {
        "epochs": config["epochs"],
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion,
        "model_save_path": config["model_save_path"],
        "cls_num_list": cls_num_list,
        "training_stage": stage
    }
    
    return full_config

def create_config(model, cls_num_list=None):
    warmup_epochs = 5
    epochs = 150
    momentum = 0.9
    weight_decay = 5e-4

    # LDAM 용 cosine annealing 함수 제작
    def cosine_annealing(epoch):
        if epoch < warmup_epochs:
            return float(epoch+1) / float(warmup_epochs)
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return scale
    
    if cls_num_list:
        learning_rate = 0.1

        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum, weight_decay=weight_decay)
        criterion = myLoss(mode="LDAM", cls_num_list=cls_num_list)
        scheduler = LambdaLR(optimizer, lr_lambda=cosine_annealing)
        model_save_path = "./models_path/Base_balanced.pth"

    else:
        learning_rate = 0.025

        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum, weight_decay=weight_decay)
        criterion = myLoss(mode="CrossEntropy")
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - (epoch/epochs))
        model_save_path = "./models_path/LDAM_unbalanced.pth"

    full_config = {
        "epochs": epochs,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion,
        "model_save_path": model_save_path,
    }

    return full_config