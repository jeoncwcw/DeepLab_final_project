# **EEE 4423 Deep Learning LAB – Final Project**

**Report:**  
[Supcon-Guided Hybrid Classification for Tail-aware Representation Learning](./Supcon-Guided%20Hybrid%20Classification%20for%20Tail-aware%20Representation%20Learning.pdf)

---

## **Overview**
 The project addresses the challenge of long‑tail data distributions and compares multiple classification models to achieve balanced performance across head, middle and tail classes.

---

## **Model Evaluation**
 A comprehensive evaluation can be performed using the Jupyter notebook `Model_tester.ipynb`. Executing this notebook allows you to load and compare several pre-trained models—namely the base model, LDAM model and a multi‑stage model. The notebook reports both absolute accuracy and relative accuracy, offering a clear, visual analysis of model performance across class groups.

---

## **Pretrained Models**
Five pre‑trained models are provided for analysis:
- `Base_balanced.pth`: The base model trained on a balanced dataset.
- `Base_unbalanced.pth`: The base model trained on an unbalanced dataset.
- `LDAM_balanced.pth`: A model trained on the balanced dataset using the LDAM strategy.
- `LDAM_unbalanced.pth`: A model trained on the unbalanced dataset using LDAM.
- `stage3_CSE.pth`: The final stage of the multi‑stage model, which incorporates a Softgate strategy and demonstrates the highest accuracy and relative accuracy.
Among these, `stage3_CSE.pth` is recognised as the best-performing model. It effectively mitigates long‑tail distribution effects and achieves balanced accuracy across all class groups.

---

## **Multi-Stage Model Architecture**
The multi‑stage model is specifically designed to cope with skewed class distributions. It proceeds through three distinct phases:
1. **Stage 1** – SupCon Loss: Encourages greater inter-class separability.
2. **Stage 2** – LDAM Loss: Places additional emphasis on underrepresented tail classes.
3. **Stage 3** – CSE Loss: Fine‑tunes the network using a balanced dataset.
Each stage employs a different loss function and data subset, progressively refining the model’s representation capability. The resulting `stage3_CSE.pth` model combines the strengths of these stages and offers the most balanced performance across the class distribution.

---

## **Usage Instructions**
1. Run the evaluation notebook: Open and execute `Model_tester.ipynb` to load the pre‑trained models and generate performance metrics.
2. Interpret results: Review the reported accuracy and relative accuracy to compare model performance. The evaluation should confirm that `stage3_CSE.pth` attains the highest metrics.

---

This project explores multiple approaches for handling long‑tail distributions and demonstrates that a multi‑stage learning strategy, culminating in the `stage3_CSE` model, yields the best balance between overall accuracy and equitable performance across class groups. The provided models and evaluation tools allow researchers to replicate and extend these findings using their own data.