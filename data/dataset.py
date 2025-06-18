def get_dataloader(mode="unbalanced", split = False):
    '''
    get_dataloader 함수는 CIFAR-100 데이터셋에서 
    mode에 따라 balanced(균형) 또는 unbalanced(불균형) 학습용 DataLoader와 
    테스트용 DataLoader를 생성하여 반환합니다.

    입력:
        mode: "balanced" 또는 "unbalanced" (기본값 "unbalanced")
        split: True이면 head, mid, tail로 분할된 DataLoader를 반환 (기본값 False)
        split이 True인 경우, head(0-33), mid(34-66), tail(67-99) 클래스로 분할하여 각각의 DataLoader 반환

    반환:
        1. split = False:
            train_loader: 학습 데이터셋 DataLoader
            test_loader:  테스트 데이터셋 DataLoader
            cls_num_list: 학습 데이터셋의 클래스별 샘플 수 리스트
        2. split = True:
            {(head_train, head_test),
            (mid_train, mid_test),
            (tail_train, tail_test)}
    '''
    import torch
    import random as rd
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    import numpy as np
    from copy import deepcopy
    
    torch.manual_seed(42); rd.seed(42); np.random.seed(42)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    ])
    full_dataset = datasets.CIFAR100(root='../CIFAR100', train=True, download=True, transform=None)

    targets = np.array(full_dataset.targets)
    class_indices = {i:np.where(targets == i)[0].tolist() for i in range(100)}

    test_indices = list()
    for i in range(100):
        indices = class_indices[i]
        selected = rd.sample(indices, 50)
        test_indices.extend(selected)
        class_indices[i] = list(set(indices) - set(selected))

    train_indices = list()

    if mode.lower() == "balanced":
        for i in range(100):
            indices = class_indices[i]
            selected = rd.sample(indices, 450)
            train_indices.extend(selected)
    else: 
        for i in range(100):
            indices = class_indices[i]
            selected = rd.sample(indices, 450 - 4*i)
            train_indices.extend(selected)
    
    train_ds = deepcopy(full_dataset)
    test_ds = deepcopy(full_dataset)
    train_ds.transform = train_tf
    test_ds.transform = test_tf

    if not split:
        train_dataset = Subset(train_ds, train_indices)
        test_dataset  = Subset(test_ds, test_indices)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

        train_targets = np.array(full_dataset.targets)[train_indices]
        cls_num_list = [np.sum(train_targets == i) for i in range(100)]

        return train_loader, test_loader, cls_num_list
    
    head_cls = range(0, 34)
    mid_cls = range(34, 67)
    tail_cls = range(67, 100)
    
    def filter_indices(indices, classes):
        targets_sel = np.array(full_dataset.targets)[indices]
        mask = np.isin(targets_sel, classes)
        return list(np.array(indices)[mask])
    
    sets = {}
    for name, cls_range in zip(['head', 'mid', 'tail'], [head_cls, mid_cls, tail_cls]):
        train_idx = filter_indices(train_indices, cls_range)
        test_idx = filter_indices(test_indices, cls_range)
        train_subset = Subset(train_ds, train_idx)
        test_subset = Subset(test_ds, test_idx)
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
        sets[name] = (train_loader, test_loader)
    return sets['head'], sets['mid'], sets['tail']


# _, _ = get_dataloader()