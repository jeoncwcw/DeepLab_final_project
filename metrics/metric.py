import torch

def top_1_metric(model, test_loader, device, use_scl):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if use_scl and isinstance(images, (list, tuple)):
                images = images[0].to(device)
            else:
                images = images.to(device)
            labels = labels.to(device)
            if hasattr(model, 'training_stage'):
                outputs = model(images, inference_mode=True)
            elif hasattr(model, 'use_scl') and model.use_scl:
                _, outputs = model(images)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.cpu()).sum()
            else:
                correct += (predicted == labels).sum()
    
    accuracy = 100 * correct.item() / total
    return accuracy

def relative_accuracy(model_t, model_v, model_b, test_loader, device, use_scl):
    '''
    relative_accuracy 함수는 세 개의 모델(model_t, model_v, model_b)에 대해 
    동일한 테스트 데이터셋에서 top-1 정확도를 각각 계산한 뒤,
    두 baseline 모델(model_v, model_b) 중 더 높은 정확도를 기준값(accuracy_u)으로 삼아,
    target 모델(model_t)의 정확도를 이 기준값으로 나눈 상대 정확도(relative accuracy)를 반환합니다.
    '''
    accuracy_t = top_1_metric(model_t, test_loader, device, use_scl)
    accuracy_v = top_1_metric(model_v, test_loader, device, use_scl)
    accuracy_b = top_1_metric(model_b, test_loader, device, use_scl)

    accuracy_u = max(accuracy_v, accuracy_b)
    assert accuracy_u > 0
    relative_acc = accuracy_t / accuracy_u

    return relative_acc

def evaluate_stage3_model(model, test_loader, device):
    """Stage3까지 완료된 모델에 대한 성능 평가 (기존 모델 구조 활용)"""
    model.eval()
    
    correct = 0
    total = 0
    ldam_correct = 0
    cse_correct = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            if isinstance(images, (list, tuple)):
                images = images[0].to(device)
            else:
                images = images.to(device)
            labels = labels.to(device)
            
            # 모델의 기존 구조 활용
            features = model.encoder(images).flatten(1)
            
            # 개별 분류기 성능 평가
            ldam_logits = model.ldam_classifier(features)
            cse_logits = model.cse_classifier(features)
            
            # 최종 예측 (모델의 inference 활용)
            final_outputs = model(images, inference_mode=True)
            
            _, final_pred = final_outputs.max(1)
            _, ldam_pred = ldam_logits.max(1)
            _, cse_pred = cse_logits.max(1)
            
            total += labels.size(0)
            correct += final_pred.eq(labels).sum().item()
            ldam_correct += ldam_pred.eq(labels).sum().item()
            cse_correct += cse_pred.eq(labels).sum().item()
    
    accuracy = 100 * correct / total
    ldam_accuracy = 100 * ldam_correct / total
    cse_accuracy = 100 * cse_correct / total
    
    print(f"전체 모델 정확도: {accuracy:.2f}%")
    print(f"LDAM 분류기 정확도: {ldam_accuracy:.2f}%")
    print(f"CSE 분류기 정확도: {cse_accuracy:.2f}%")
    
    return accuracy, ldam_accuracy, cse_accuracy