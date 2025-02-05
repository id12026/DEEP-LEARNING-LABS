# LAB2_MODULE 2
# Customer Churn Prediction using ANN

This repository contains two implementations of an **Artificial Neural Network (ANN)** for predicting customer churn.

## 📌 Overview
Customer churn prediction is an essential task for businesses to retain customers and reduce revenue loss. I have implement **two versions of an ANN** using TensorFlow/Keras:

1. **Baseline Model (Simple ANN)**
2. **Regularized Model (with L2 Regularization, Dropout & Batch Normalization)**

---

## 🚀 Dataset
We use a customer churn dataset (`Churn_Modelling.csv`) containing the following key features:
- **Categorical Variables**: `Geography`, `Gender`
- **Numerical Variables**: `CreditScore`, `Age`, `Balance`, etc.
- **Target Variable**: `Exited` (1 = Churn, 0 = Retained)

---

## 📂 Project Structure

```bash
├── baseline_model.py  # ANN with only input and output layers
├── regularized_model.py  # ANN with L2 Regularization, Dropout & BatchNorm
├── churn_data.csv  # Sample dataset (not included here)
├── README.md  # This file
```

---

## 📌 1: Baseline ANN (No Regularization)
This simple ANN contains:
- **1 Input Layer**
- **1 Output Layer**
- **Sigmoid Activation Function** for binary classification

### **🔹 Code Snippet**
```python
# Build ANN model (Baseline)
model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))
])
```
✅ **Optimizer:** Adam  
✅ **Loss Function:** Binary Crossentropy  
✅ **Metric:** Accuracy  

### **🔹 Performance**
- No overfitting control  
- Might struggle with complex patterns in data  

---

## 📌 : Regularized ANN (L2, Dropout & BatchNorm)
This model is improved with:
- **L2 Regularization (`l2(0.01)`)** to reduce overfitting  
- **Dropout (`Dropout(0.3)`)** to randomly deactivate neurons  
- **Batch Normalization** to stabilize learning  

### **🔹 Code Snippet**
```python
# Build ANN model with Regularization
model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    BatchNormalization()
])
```
✅ **Prevents overfitting**  
✅ **Stabilizes learning & improves generalization**  

---

## 📊 Performance Comparison

| Model  | Regularization | Accuracy  | Overfitting Risk |
|--------|--------------|-----------|-----------------|
| Baseline ANN | ❌ None | Moderate | High |
| Regularized ANN | ✅ L2, Dropout, BatchNorm | Higher | Low |

---

## 📈 Visualizations
### **🔹 Confusion Matrix**
Both models generate a **confusion matrix** to evaluate predictions:

![Confusion Matrix](confusion_matrix.png)

### **🔹 Accuracy & Loss Graphs**
Models also plot:
1. **Training vs. Validation Accuracy**
2. **Training vs. Validation Loss**

---

## 🛠️ Installation & Usage
### **📌 1️⃣ Install Dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras
```

### **📌 2️⃣ Run the Models**
- **Run the Baseline Model**
  ```bash
  python baseline_model.py
  ```
- **Run the Regularized Model**
  ```bash
  python regularized_model.py
  ```

---

## 🏆 Results & Insights
1. The **regularized model** performs better than the baseline model.
2. **Dropout & L2 regularization** help prevent overfitting.
3. **Batch Normalization** stabilizes the training process.

---


🔗 **Developed by:** [Mohitha Bandi_22WU0105037_DS-B](https://github.com/id12026)  
📧 **E-mail:** mohitha12026@gmail.com
```

---

### **📌 Key Features of this README**
✅ **Well-structured & clear**  
✅ **Explains both models**  
✅ **Shows performance comparison**  
✅ **Includes installation & usage steps**  


Would you like me to customize it further? 🚀
