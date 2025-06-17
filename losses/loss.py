import torch.nn as nn

def myLoss(mode = "CrossEntropy"):
    if mode == "CrossEntropy":
        return nn.CrossEntropyLoss()