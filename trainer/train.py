import torch
import numpy as np
import torch.nn.functional as F

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
        if test_loss < best_loss:
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
        
        model.set_training_stage(stage)
        config = create_stage_config(stage, cls_num_list, model)

        epochs = config["epochs"]
        optimizer = config["optimizer"]
        scheduler = config["scheduler"]
        criterion = config["criterion"]
        model_save_path = config["model_save_path"]

        best_loss = float('inf')

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            for data, labels in train_loader:
                labels = labels.to(device)
                optimizer.zero_grad()
                
                match stage:
                    case 1:
                        images = torch.cat([data[0], data[1]], dim=0).to(device)
                        scl_features = model(images)
                        f1, f2 = torch.chunk(scl_features, 2, dim=0)
                        loss_scl = criterion(torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1), labels)
                        loss_scl.backward(retain_graph=True)
                        total_loss += loss_scl.item()
                    case 2:
                        if isinstance(data, (list, tuple)):
                            images = data[0].to(device)
                        else:
                            images = data.to(device)
                        logits = model(images)
                        loss_ldam = criterion(logits, labels)
                        loss_ldam.backward()
                        total_loss += loss_ldam.item()
                    case 3:
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
                if test_loss < best_loss:
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