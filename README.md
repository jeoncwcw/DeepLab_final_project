# 🌟 DeepLab_final_project
EEE4423 DeeplearningLab Final Project

---
📄 **리포트 링크:**  
[Supcon-Guided Hybrid Classification for Tail-aware Representation Learning](./Supcon-Guided%20Hybrid%20Classification%20for%20Tail-aware%20Representation%20Learning.pdf)

---

## 🧪 **Model Tester**
`Model_tester.ipynb`를 실행하면 다양한 모델을 검증할 수 있습니다.  
이 파일은 **Base Model**, **LDAM Model**, **Multi-Stage Model**을 포함하여 각 모델의 성능을 평가하고 비교할 수 있는 기능을 제공합니다.  
Accuracy와 Relative Accuracy를 기준으로 모델의 성능을 시각적으로 분석할 수 있습니다.

---

## 🏆 **Best Model: `stage3_CSE.pth`**
`stage3_CSE.pth`는 Multi-Stage Model의 최종 단계에서 학습된 모델로, 가장 높은 **Accuracy**와 **Relative Accuracy**를 기록한 모델입니다.  
이 모델은 **Long-tail Distribution 문제**를 효과적으로 해결하며, **Head/Middle/Tail 클래스 그룹**에서 균형 잡힌 성능을 보여줍니다.

---

### 📂 **모델 파일 설명**
- **`Base_balanced.pth`**: Balanced 데이터셋에서 학습된 Base Model.
- **`Base_unbalanced.pth`**: Unbalanced 데이터셋에서 학습된 Base Model.
- **`LDAM_balanced.pth`**: Balanced 데이터셋에서 LDAM 전략을 사용해 학습된 모델.
- **`LDAM_unbalanced.pth`**: Unbalanced 데이터셋에서 LDAM 전략을 사용해 학습된 모델.
- **`stage3_CSE.pth`**: Multi-Stage Model의 최종 단계에서 학습된 모델로, Softgate Strategy를 포함하여 가장 높은 성능을 기록한 모델.

---

## 🔧 **Multi-Stage Model 구현**
Multi-Stage Model은 **Long-tail Distribution 문제**를 해결하기 위해 설계된 모델로, 다음과 같은 단계로 구성됩니다:

1️⃣ **Stage 1**: SupCon Loss를 사용하여 클래스 간의 구분을 강화.  
2️⃣ **Stage 2**: LDAM Loss를 사용하여 Tail 클래스의 중요도를 높임.  
3️⃣ **Stage 3**: Balanced 데이터셋을 사용하여 CSE Loss로 모델을 Fine-tuning.  

각 단계는 서로 다른 손실 함수와 데이터셋을 사용하여 모델의 성능을 점진적으로 향상시킵니다.  
최종적으로 `stage3_CSE.pth`는 모든 단계를 거쳐 학습된 **최상의 모델**입니다.

---

## 🚀 **실행 방법**
1. `Model_tester.ipynb`를 열고 실행하여 다양한 모델의 성능을 검증합니다.
2. 각 모델의 **Accuracy**와 **Relative Accuracy**를 확인하고, `stage3_CSE.pth`가 가장 높은 성능을 기록했음을 확인합니다.

---

이 프로젝트는 **Long-tail Distribution 문제**를 해결하기 위한 다양한 접근법을 탐구하며, **Multi-Stage Model**이 가장 효과적인 해결책임을 보여줍니다.  
✨ **최적의 모델을 통해 균형 잡힌 성능을 달성하세요!**