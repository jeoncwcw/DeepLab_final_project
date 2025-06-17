import torch

def top_1_metric(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.cpu()).sum()
            else:
                correct += (predicted == labels).sum()
    
    accuracy = 100 * correct.item() / total
    return accuracy

def relative_accuracy(model_t, model_v, model_b, test_loader, device):
    '''
    relative_accuracy 함수는 세 개의 모델(model_t, model_v, model_b)에 대해 
    동일한 테스트 데이터셋에서 top-1 정확도를 각각 계산한 뒤,
    두 baseline 모델(model_v, model_b) 중 더 높은 정확도를 기준값(accuracy_u)으로 삼아,
    target 모델(model_t)의 정확도를 이 기준값으로 나눈 상대 정확도(relative accuracy)를 반환합니다.
    '''
    accuracy_t = top_1_metric(model_t, test_loader, device)
    accuracy_v = top_1_metric(model_v, test_loader, device)
    accuracy_b = top_1_metric(model_b, test_loader, device)

    accuracy_u = max(accuracy_v, accuracy_b)
    assert accuracy_u > 0
    relative_acc = accuracy_t / accuracy_u

    return relative_acc