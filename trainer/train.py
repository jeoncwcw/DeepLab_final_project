import torch
import numpy as np
import torch.nn as nn
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

def get_dynamic_scl_alpha(epoch, total_epochs):
    # 처음에는 약하게, 후반에 강하게
    start_alpha = 0.05
    end_alpha = 0.3
    progress = min(1.0, epoch / (total_epochs * 0.7))
    return start_alpha + progress * (end_alpha - start_alpha)


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
                        start_stage=1, load_stage1_path=None, load_stage_2_path=None):
    from configs._config_ import create_stage_config

    train_losses = list()
    test_losses = list()


    # Pre-trained model 로드
    if start_stage >= 2 and load_stage1_path:
        print(f"=== Loading Stage 1 model from {load_stage1_path} ===")
        model.load_state_dict(torch.load(load_stage1_path))
        print("Stage 1 model loaded successfully!")
        
    if start_stage >= 3 and load_stage_2_path:
        print(f"=== Loading Stage 2 model from {load_stage_2_path} ===")
        model.load_state_dict(torch.load(load_stage_2_path))
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
        criterion_scl = config["criterion_scl"]
        model_save_path = config["model_save_path"]
        use_scl = config["use_scl"]

        best_loss = float('inf')

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            total_loss_scl = 0.0
            total_loss_ldam = 0.0

            for data, labels in train_loader:
                labels = labels.to(device)
                optimizer.zero_grad()

                if stage == 1:
                    if isinstance(data, (list, tuple)):
                        images = data[0].to(device)
                    else:
                        images = data.to(device)
                    logits = model(images)                    
                    loss = criterion(logits, labels)
                    loss.backward()
                elif stage == 2:
                    images = torch.cat([data[0], data[1]], dim=0).to(device)
                    scl_features, ldam_logits = model(images)
                    f1, f2 = torch.chunk(scl_features, 2, dim=0)
                    loss_scl = criterion_scl(torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1), labels)
                    logits_for_ldam, _ = torch.chunk(ldam_logits, 2, dim=0)
                    loss_ldam = criterion(logits_for_ldam, labels)
                    scl_alpha = get_dynamic_scl_alpha(epoch, epochs)
                    ldam_lambda = config["ldam_lambda"]
                    beta = 0.2
                    (scl_alpha*loss_scl).backward(retain_graph=True)

                    saved_grads = []
                    for p in model.encoder.parameters():
                        saved_grads.append(p.grad.detach().clone() if p.grad is not None else None)
                        if p.grad is not None:
                            p.grad.zero_()
                    (ldam_lambda * loss_ldam).backward()

                    with torch.no_grad():
                        for p, g_scl in zip(model.encoder.parameters(), saved_grads):
                            if g_scl is not None and p.grad is not None:
                                p.grad = g_scl + beta * p.grad

                    total_loss += scl_alpha * loss_scl.item() + ldam_lambda * loss_ldam.item()
                    total_loss_scl += loss_scl.item()
                    total_loss_ldam += loss_ldam.item()

                elif stage == 3:
                    if isinstance(data, (list, tuple)):
                        images = data[0].to(device)
                    else:
                        images = data.to(device)
                    cse_logits = model(images)
                    loss = criterion(cse_logits, labels)
                    loss.backward()
                
                
                optimizer.step()
                if stage in [1,3]:
                    total_loss += loss.item()
            scheduler.step()
            test_loss = test_loss_cal(model, test_loader, criterion, device)
            avg_train_loss = total_loss / len(train_loader)
            if stage == 2:
                avg_ldam_loss = total_loss_ldam / len(train_loader)
                avg_scl_loss = total_loss_scl / len(train_loader)
                print(f"[Epoch {epoch+1}/{epochs}] Total: {avg_train_loss:.4f} | SCL: {avg_scl_loss:.4f} | LDAM: {avg_ldam_loss:.4f} | Test: {test_loss:.4f}")
            else:
                    print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f}")

            train_losses.append(avg_train_loss)
            test_losses.append(test_loss)
                
            # Best model 저장
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), model_save_path)
                print(f"[Best Model Saved!] Stage {stage}")
        
        print(f"Stage {stage} completed. Model saved to {model_save_path}")

        if stage < 3:
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