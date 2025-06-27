# 🌾 Crop Recommendation System

A machine learning-based system to predict the most suitable crop to grow based on environmental parameters like nitrogen, phosphorus, potassium levels, temperature, humidity, pH, and rainfall.

---

## 📌 Project Overview

This project leverages supervised machine learning models to classify and recommend crops based on real-time soil and weather conditions. It's ideal for modern agriculture solutions where farmers can make data-driven decisions.

---

## 🧠 Algorithms Used

The following models were trained and evaluated:

- Naive Bayes
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest ✅ (Best performer)

---

## ✅ Model Accuracy

| Algorithm               | Accuracy   |
|------------------------|------------|
| Naive Bayes            | 99.39%     |
| SVM                    | 96.36%     |
| KNN                    | 97.73%     |
| Decision Tree          | 98.33%     |
| Random Forest ✅        | **99.24%** |

---

## 📊 Dataset

- **File:** `crop_recommendation.csv`
- **Samples:** 2200
- **Columns:**
  - `N` (Nitrogen)
  - `P` (Phosphorus)
  - `K` (Potassium)
  - `temperature`
  - `humidity`
  - `ph`
  - `rainfall`
  - `label` (crop name)

The dataset is clean, balanced, and contains no missing or duplicate values.

---

## 🧹 Preprocessing Steps

- Label encoding (`label` → `label_num`)
- Train-test split (70%-30%)
- Standardization using `StandardScaler`

---

## 🚀 Prediction Function

A reusable function for prediction:

```python
def predict_crop(N, P, K, temperature, humidity, pH, rainfall):
    input_values = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    prediction = model.predict(input_values)
    return prediction[0]
```

### 🧪 Example Prediction

```python
predict_crop(7, 6, 7, 25.67, 50.24, 8.96, 88.88)
# Output: "ORANGE is the best crop to be cultivated right there"
```

---

## 📦 Requirements

```bash
numpy
pandas
seaborn
matplotlib
scikit-learn
```

Install them with:

```bash
pip install <requirement>
```

---

## 💻 How to Run

1. Clone this repo
2. Place `crop_recommendation.csv` in the working directory
3. Run the script:
```bash
python crop_prediction.py
```

---

## 📈 Visualizations

- Distribution plots using Seaborn
- Correlation matrix
- Accuracy comparison among models

---

## 📚 References

- [Kaggle - Crop Recommendation Dataset](https://www.kaggle.com/)
- [Scikit-learn Docs](https://scikit-learn.org/)
- [Seaborn Docs](https://seaborn.pydata.org/)

---

## 📞 Contact Me
Feel free to reach out to me via email at bhuvani1102@gmail.com or connect with me on LinkedIn at https://www.linkedin.com/in/bhuvani1102
