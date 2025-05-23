# ITRI616-Diabetes-Classification
A classification project using machine learning to predict family history of diabetes

---

#  ITRI616 - Diabetes Classification Project

##  Project Overview

This repository contains the full implementation of a machine learning classification project conducted as part of the **Artificial Intelligence 1 (ITRI616)** module at **North-West University (Vaal Triangle Campus)**. The objective was to build a predictive system capable of determining whether an individual has a **family history of diabetes** based on a mix of demographic, lifestyle, and physiological factors.

The project follows a full ML pipeline — from dataset selection and preprocessing to model training, evaluation, and reporting — and reflects a realistic approach to healthcare data analysis.

---

##  Dataset Description

- **Source:** Kaggle  
  [Diabetes Prediction Dataset by MarshalPatel3558](https://www.kaggle.com/datasets/marshalpatel3558/diabetes-prediction-dataset)
- **License:** MIT (open for academic use)
- **Instances:** 10,000 rows  
- **Features:** 20 (including age, BMI, cholesterol levels, smoking status, etc.)
- **Target Variable:** `Family_History_of_Diabetes` (binary classification)

The dataset reflects real-world complexity, including missing values, class imbalance, and mixed feature types.

---

##  ML Algorithms Implemented

Four classification algorithms were developed and compared:

-  Decision Tree  
-  Random Forest  
-  Gradient Boosting Classifier  
-  Support Vector Machine (SVM)

---

##  Model Development Process

### 1. **Preprocessing & Feature Engineering**
- Missing value imputation (mode for categorical fields)
- One-hot encoding for multi-class categorical features
- Standardization using `StandardScaler` (for SVM)
- Class balancing using **SMOTE** (Synthetic Minority Oversampling)

### 2. **Train-Test Split**
- Stratified 80/20 split to preserve class distribution

### 3. **Model Training**
- Models were trained using default scikit-learn parameters
- No hyperparameter tuning was performed at this stage

---

##  Evaluation Metrics

Each model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC
- Confusion Matrix
- Precision-Recall Curve

###  Best Performing Model:
- **Gradient Boosting Classifier** achieved highest ROC AUC and F1-score.

See `figures/` folder for visualizations (e.g., ROC curves, PR curves).

---

##  Repository Structure

```

ITRI616-Diabetes-Classification/
│
├── notebook/
│   └── Diabetes-616.ipynb
│
├── data/
│   └── diabetes\_dataset.csv
│
├── figures/
│   └── figure11\_precisionrecall.png
│   └── \[other visuals]
│
├── Project\_Report.docx
├── README.md
├── requirements.txt
├── EnvironmentDetails.md

````

---

##  How to Run the Project

### 1. Clone the repository:
```bash
git clone https://github.com/monica-serwala/ITRI616-Diabetes-Classification.git
cd ITRI616-Diabetes-Classification
````

### 2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # (or venv\Scripts\activate on Windows)
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Launch the notebook:

```bash
jupyter notebook
```

---

## 🖥️ Environment Configuration

See `EnvironmentDetails.md` for full computational setup, including:

* Python 3.11
* scikit-learn
* pandas, numpy, matplotlib, seaborn, imblearn
* OS: Windows 11, Intel Core i5, 8 GB RAM

---

##  License

This repository is submitted for academic purposes only. Dataset used under MIT license from Kaggle.

---

##  Author

**Monica Mmakokong Serwala**
Final-Year Information Technology Student
North-West University (Vaal Triangle Campus)
ITRI616: Artificial Intelligence 1
Lecturer: Mr. Melvin Kisten

---

##  Contact

For academic queries: \[Insert your NWU email if allowed]

```
