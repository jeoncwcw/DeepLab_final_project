import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from losses.loss import myLoss

def create_stage_scheduler(stage, epochs):
    warmup_epochs = 5
    if stage == 1:
        def stage1_scheduler(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            else:
                progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        return stage1_scheduler
    
    elif stage == 2:
        def stage2_scheduler(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            else:
                progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
                return 0.3 + 0.4 * (1.0 + math.cos(math.pi * progress))
        return stage2_scheduler
    
    else:
        def stage3_scheduler(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            else:
                return 1.0
        return stage3_scheduler
    
def get_stage_configs():
    return {
        "stage1": {
            "epochs": 70,
            "lr": 0.05,
            "use_scl": False,
            "scl_alpha": 0,
            "ldam_lambda": 0.5,
            "model_save_path": "./models_path/stage1_LDAM_only.pth"
        },
        "stage2": {
            "epochs": 40,
            "lr": 0.015,
            "use_scl": True,
            "scl_alpha": 0.3,
            "ldam_lambda": 0.8,
            "model_save_path": "./models_path/stage2_LDAM_SCL.pth"
        },
        "stage3": {
            "epochs": 30, 
            "lr": 0.005,
            "use_scl": False,
            "scl_alpha": 0,
            "ldam_lambda": 0,
            "model_save_path": "./models_path/stage3_CSE_final.pth"
        }
    }

def create_stage3_optimizer(model, config):
    base_lr = config["lr"]

    encoder_children = list(model.encoder.children())
    frozen_params = []
    tunable_params = []

    for i, child in enumerate(encoder_children):
        if i < len(encoder_children) - 2:
            frozen_params.extend(list(child.parameters()))
        else:
            tunable_params.extend(list(child.parameters()))

    for param in frozen_params:
        param.requires_grad = False
    
    optimizer = torch.optim.SGD([
        {'params': tunable_params, 'lr': base_lr * 0.1},   
        {'params': model.cse_classifier.parameters(), 'lr': base_lr}
    ], momentum=0.9, weight_decay=5e-4)
    
    return optimizer

def create_stage_config(stage, cls_num_list, model):
    stage_configs = get_stage_configs()
    config = stage_configs[f"stage{stage}"]

    if stage == 3:
        criterion = myLoss(mode="CrossEntropy")
        criterion_scl = None
    else:
        criterion = myLoss(mode="LDAM", cls_num_list=cls_num_list)
        criterion_scl = myLoss(mode="SCL", cls_num_list=cls_num_list) if config["use_scl"] else None

    # if stage == 3:
    #     optimizer = create_stage3_optimizer(model, config)
    # else:
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=5e-4)

    scheduler = LambdaLR(optimizer, lr_lambda=create_stage_scheduler(stage, config["epochs"]))

    full_config = {
        "epochs": config["epochs"],
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion,
        "model_save_path": config["model_save_path"],
        "use_scl": config["use_scl"],
        "scl_alpha": config["scl_alpha"],
        "ldam_lambda": config["ldam_lambda"],
        "criterion_scl": criterion_scl,
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
        model_save_path = "./models_path/Base_unbalanced.pth"

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