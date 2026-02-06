# Fraud Detection using Machine Learning

## 📌 Overview
This project predicts the probability of fraudulent transactions using both **supervised** and **unsupervised** approaches.  
The dataset is highly imbalanced, making it a realistic challenge for fraud detection systems.

- **Supervised:** Logistic Regression  
- **Unsupervised:** Isolation Forest (anomaly detection)  

---

## 📌 Dataset
- Kaggle Credit Card Fraud dataset  
- **Total transactions:** 284,807  
- **Fraud rate:** 0.17%  
- Features are anonymized (`V1`–`V28`) + `Amount`  
- Target column: `Class` (0 = normal, 1 = fraud)  

---

## 📌 Approach

1. **Data Exploration & Preprocessing**  
   - Checked for missing values, handled NaNs  
   - Scaled features using `StandardScaler`  
   - Split data into **train/test sets** with stratification

2. **Supervised Model: Logistic Regression**  
   - Trained with `class_weight='balanced'` to handle class imbalance  
   - Cross-validation for reliable PR-AUC (~0.92)   
   - Evaluated with Precision, Recall, F1-score, PR-AUC, Confusion Matrix

3. **Unsupervised Model: Isolation Forest**  
   - Trained on **normal transactions only**  
   - Detected anomalies as potential frauds  
   - Accuracy is high (dominated by normal transactions), but PR-AUC is low (~0.13)  
   - Demonstrates unsupervised anomaly detection for cases without labels  

4. **Evaluation Metrics**  
   - **Precision-Recall curve**: main metric for imbalanced datasets  
   - Confusion matrices for class-wise performance  
   - Comparison of supervised vs unsupervised approaches  

---

## 📌 Results

| Model                  | Precision | Recall | F1-Score | PR-AUC |
|------------------------|----------|--------|----------|--------|
| Logistic Regression    | 0.83     | 1.0    | 0.90     | 0.92   |
| Isolation Forest       | 0.14     | 0.20  | 0.16     | 0.13   |

> Notes:  
> - Logistic Regression (supervised) performs much better due to labeled data.  
> - Isolation Forest (unsupervised) highlights anomalies but struggles to detect all frauds in extremely imbalanced data.  
> - Accuracy is misleading for highly imbalanced datasets; PR-AUC and recall are more meaningful.

---

## 📌 Key Takeaways
- **Supervised ML models** are most effective when labeled data is available.  
- **Unsupervised anomaly detection** is useful when labels are missing, but limited in precision.  
- **Cross-validation** ensures performance metrics are realistic and not over-optimistic.  

---

## 📌 How to Run
1. Clone the repository  
2. Install dependencies:

```bash
pip install -r requirements.txt
