import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

def test_loss_cal(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    count = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            if hasattr(model, 'training_stage'):
                if model.training_stage == 2:
                    output = model(images)
                    if isinstance(output, tuple):
                        _, outputs = output
                    else:
                        outputs = output
                else:
                    outputs = model(images)
            else:
                outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            count += 1
    test_loss = test_loss / count
    model.train()
    return test_loss

def create_finetune_loader(train_loader, cls_num_list):
    print("=== Creating a new balanced dataloader for fine-tuning ===")
    tail_class_start_index = 67
    avg_tail_count = int(np.mean(cls_num_list[tail_class_start_index:]))
    new_balanced_indices = list()
    original_train_indices = train_loader.dataset.indices
    all_targets = np.array(train_loader.dataset.dataset.targets)

    indices_by_class = {i: [] for i in range(100)}
    for idx in original_train_indices:
        label = all_targets[idx]
        indices_by_class[label].append(idx)

    for i in range(100):
        indices_for_class_i = indices_by_class[i]
        if i < tail_class_start_index:
            sampled_indices = np.random.choice(indices_for_class_i, avg_tail_count, replace=False)
            new_balanced_indices.extend(sampled_indices)
        else:
            new_balanced_indices.extend(indices_for_class_i)

    new_subset = Subset(train_loader.dataset.dataset, new_balanced_indices)

    import platform
    num_workers=0 if platform.system() == 'Windows' else train_loader.num_workers
    new_train_loader = DataLoader(
        new_subset,
        batch_size = train_loader.batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    print("=== Done ===\n")
    return new_train_loader


def compute_intra_class_variance(features, labels, num_classes=100):
        variances = []
        for c in range(num_classes):
            class_mask = (labels == c)
            if class_mask.sum() > 1:
                class_features = features[class_mask]
                class_center = class_features.mean(dim=0)
                variance = ((class_features - class_center) ** 2).sum(dim=1).mean()
                variances.append(variance.item())
        return np.mean(variances)

def compute_inter_class_distance(features, labels, num_classes=100):
    class_centers = []
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            class_feat = features[mask]
            center = class_feat.mean(dim=0)
            class_centers.append(center)
    class_centers = torch.stack(class_centers, dim=0)  # (C, D)
    # 모든 클래스 중심 간 거리 평균
    return torch.pdist(class_centers, p=2).mean().item()


def myTrainer(model, train_loader, test_loader, device, config):
    epochs = config["epochs"]
    optimizer = config["optimizer"] 
    scheduler = config["scheduler"]
    criterion = config["criterion"]
    model_save_path = config["model_save_path"]
    train_losses = list()
    test_losses = list()    
    best_loss = float('inf')

    print("=== Single Loss Training Start ===")
    model.train()
    for epoch in range(epochs):
        if hasattr(criterion, 'update_epoch'):
                criterion.update_epoch(epoch, epochs)
        total_loss = 0.0
        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)
                
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()                
            total_loss += loss.item()
            
        scheduler.step()
        test_loss = test_loss_cal(model, test_loader, criterion, device)
        print(f"[epoch {epoch+1}] train_loss: {total_loss/len(train_loader):.4f}, test_loss: {test_loss:.4f}")
        train_losses.append(total_loss/len(train_loader))
        test_losses.append(test_loss)
        if test_loss < best_loss and epoch > 75:
            best_loss = test_loss
            torch.save(model.state_dict(), model_save_path)
            print("[Best Model Saved!]")
    return train_losses, test_losses

def three_stage_trainer(model, train_loader, test_loader, device, cls_num_list,
                        start_stage=1, load_stage1_path=None, load_stage2_path=None):
    from configs._config_ import create_stage_config

    train_losses = list()
    test_losses = list()

    # Pre-trained model 로드
    if start_stage >= 2 and load_stage1_path:
        print(f"=== Loading Stage 1 model from {load_stage1_path} ===")
        model.load_state_dict(torch.load(load_stage1_path))
        print("Stage 1 model loaded successfully!")
        
    if start_stage >= 3 and load_stage2_path:
        print(f"=== Loading Stage 2 model from {load_stage2_path} ===")
        model.load_state_dict(torch.load(load_stage2_path))
        print("Stage 2 model loaded successfully!")

    # Stage 실행
    stages_to_run = list(range(start_stage, 4))  # start_stage부터 3까지

    for stage in stages_to_run:
        print(f"\n === Stage{stage} Training Start ===")
        if stage == 3:
            train_loader = create_finetune_loader(train_loader, cls_num_list)
        model.set_training_stage(stage)
        config = create_stage_config(stage, cls_num_list, model)

        epochs = config["epochs"]
        optimizer = config["optimizer"]
        scheduler = config["scheduler"]
        criterion = config["criterion"]
        model_save_path = config["model_save_path"]

        best_loss = float('inf')

        for epoch in range(epochs):
            if hasattr(criterion, 'update_epoch'):
                criterion.update_epoch(epoch, epochs)
            model.train()
            total_loss = 0.0

            for data, labels in train_loader:
                labels = labels.to(device)
                optimizer.zero_grad()
                
                if stage == 1:
                        images = torch.cat([data[0], data[1]], dim=0).to(device)
                        scl_features = model(images)
                        f1, f2 = torch.chunk(scl_features, 2, dim=0)
                        loss_scl = criterion(torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1), labels)
                        loss_scl.backward(retain_graph=True)
                        total_loss += loss_scl.item()
                elif stage == 2:
                        if isinstance(data, (list, tuple)):
                            images = data[0].to(device)
                        else:
                            images = data.to(device)
                        logits = model(images)
                        loss_ldam = criterion(logits, labels)
                        loss_ldam.backward()
                        total_loss += loss_ldam.item()
                elif stage == 3:
                        if isinstance(data, (list, tuple)):
                            images = data[0].to(device)
                        else:
                            images = data.to(device)
                        cse_logits = model(images)
                        loss_cse = criterion(cse_logits, labels)
                        loss_cse.backward()
                        total_loss += loss_cse.item()          
                optimizer.step()
            scheduler.step()
            if stage == 1:
                print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {total_loss / len(train_loader):.4f}")
                with torch.no_grad():
                    features = []
                    labels_list = []
                    for images, labels in test_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        features.append(F.normalize(model.projection_head(model.encoder(images).flatten(1)), dim=1).cpu())
                        labels_list.append(labels.cpu())
                    features = torch.cat(features)
                    labels_list = torch.cat(labels_list)
                    intra_class_variance = compute_intra_class_variance(features, labels_list)
                    inter_class_distance = compute_inter_class_distance(features, labels_list)
                    inter_intra_ratio = inter_class_distance / intra_class_variance
                    print(f"Intra-class Variance: {intra_class_variance:.4f} | Inter-class Distance: {inter_class_distance:.4f}")
                    print(f"Ratio: {inter_intra_ratio:.4f}")
            else:
                test_loss = test_loss_cal(model, test_loader, criterion, device)
                print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {total_loss / len(train_loader):.4f} | Test Loss: {test_loss:.4f}")
            
            train_losses.append(total_loss / len(train_loader))
            if stage == 1:
                test_losses.append(intra_class_variance)
            else:
                test_losses.append(test_loss)
                
            # Best model 저장
            if stage == 1:
                pass
            else:
                if test_loss < best_loss and epoch > 36:
                    best_loss = test_loss
                    torch.save(model.state_dict(), model_save_path)
                    print(f"[Best Model Saved!] Stage {stage}")
        if stage == 1:
            torch.save(model.state_dict(), model_save_path)
        print(f"Stage {stage} completed. Model saved to {model_save_path}")

        if stage != 1:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for data, target in test_loader:
                    if isinstance(data, (list, tuple)):
                        data = data[0]
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    if isinstance(output, tuple):
                        output = output[1]  # logits만 선택
                    pred = output.argmax(dim=1)
                    total += target.size(0)
                    correct += pred.eq(target).sum().item()
                acc = correct / total
                print(f"Stage {stage} Accuracy: {acc:.4f}")
            model.train()
    
    return train_losses, test_losses