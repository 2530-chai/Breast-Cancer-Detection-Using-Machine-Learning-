# 🧠 Breast Cancer Prediction Using Machine Learning

This project aims to build machine learning models to accurately classify breast cancer diagnoses (Malignant or Benign) based on a dataset of medical imaging features.

---

## 📌 Objective

To predict whether a breast tumor is malignant or benign using statistical and machine learning techniques, enabling early detection and potential medical intervention.

---

## 🗂️ Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Size**: 569 rows × 33 columns
- **Features**:
  - Tumor characteristics like `radius_mean`, `texture_mean`, `area_worst`, etc.
  - Diagnosis labels: `M` = Malignant, `B` = Benign

---

## 🔍 Steps Performed

### 1. Data Preprocessing
- Dropped irrelevant columns (`id`, `Unnamed: 32`)
- Converted categorical diagnosis labels into binary (0 for B, 1 for M)
- Standardized/normalized numerical features

### 2. Exploratory Data Analysis (EDA)
- Checked data distribution and correlation
- Used heatmaps, histograms, and boxplots for visual analysis
- Identified top contributing features

### 3. Model Building & Evaluation
- Models used:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Random Forest Classifier
- Evaluated using:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix

---

## 🧪 Results

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression| 97.4%    |
| SVM                | 98.2%    |
| KNN                | 96.5%    |
| Random Forest      | 98.2%    |

- Top features influencing prediction:
  - `radius_mean`
  - `concavity_mean`
  - `area_mean`
  - `concave points_worst`

---

## 🛠️ Tools Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## 📚 Future Improvements

- Hyperparameter tuning with GridSearchCV
- Feature reduction via PCA
- Deploy as a web application with Flask or Streamlit

---

## 📁 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/breast-cancer-prediction.git
cd breast-cancer-prediction

# (Optional) Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebook
jupyter notebook
