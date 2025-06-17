def get_dataloader(mode="unbalanced"):
    '''
    get_dataloader 함수는 CIFAR-100 데이터셋에서 
    mode에 따라 balanced(균형) 또는 unbalanced(불균형) 학습용 DataLoader와 
    테스트용 DataLoader를 생성하여 반환합니다.

    입력:
        mode: "balanced" 또는 "unbalanced" (기본값 "unbalanced")

    반환:
        train_loader: 학습 데이터셋 DataLoader
        test_loader:  테스트 데이터셋 DataLoader
    '''
    import torch
    import random as rd
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    import numpy as np
    
    torch.manual_seed(42); rd.seed(42); np.random.seed(42)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    full_dataset = datasets.CIFAR100(root='../CIFAR100', train=True, download=True, transform=transform)

    targets = np.array(full_dataset.targets)
    class_indices = {i:np.where(targets == i)[0].tolist() for i in range(100)}

    test_indices = list()
    for i in range(100):
        indices = class_indices[i]
        selected = rd.sample(indices, 50)
        test_indices.extend(selected)
        class_indices[i] = list(set(indices) - set(selected))

    train_indices = list()

    if mode == "balanced":
        for i in range(100):
            indices = class_indices[i]
            selected = rd.sample(indices, 450)
            train_indices.extend(selected)
    else: 
        for i in range(100):
            indices = class_indices[i]
            selected = rd.sample(indices, 450 - 4*i)
            train_indices.extend(selected)
    
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset  = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

    return train_loader, test_loader
    
