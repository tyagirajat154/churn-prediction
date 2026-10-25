<div align="center">

# Customer Churn Prediction

### Machine Learning | Logistic Regression from Scratch

<p>
  A machine learning project that predicts customer churn using Python and NumPy.
</p>

</div>

---

## 📌 About the Project

Customer churn occurs when customers stop using a company's services. Predicting churn helps businesses identify customers who may leave and improve their retention strategies.

This project focuses on building a **Logistic Regression model from scratch using NumPy**, providing a practical understanding of the core concepts behind machine learning classification.

## 🎯 Project Objectives

* Predict whether a customer is likely to leave a telecommunications service.
* Implement Logistic Regression from scratch using NumPy.
* Understand the Sigmoid function and Gradient Descent.
* Perform data preprocessing and feature scaling.
* Evaluate model performance on unseen data.

## 📊 Dataset

The project uses the **IBM Telco Customer Churn dataset**, containing approximately 7,043 customer records.

The dataset includes customer demographics, subscribed services, account information, and churn status.

**Target Variable:** `Churn`

* `Yes` — Customer left the service.
* `No` — Customer continued using the service.

## 🛠️ Tech Stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
</p>

## 🧠 Machine Learning Approach

The project implements Logistic Regression from scratch using NumPy.

Key concepts include:

* Data preprocessing
* Feature scaling
* Sigmoid activation function
* Gradient Descent optimization
* Model training and prediction
* Test-set evaluation

Scikit-learn is used for supporting tasks such as data splitting and scaling, where applicable. The Logistic Regression algorithm itself is implemented using NumPy.

## ⚙️ How It Works

1. Load the customer churn dataset.
2. Explore and preprocess the data.
3. Prepare features and the target variable.
4. Split the data into training and testing sets.
5. Scale the features.
6. Train the custom Logistic Regression model using Gradient Descent.
7. Generate predictions and evaluate performance.

## 📈 Results

The current implementation achieved approximately **80% test accuracy**.

*The reported accuracy should be verified against the final notebook.*

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/tyagirajat154/churn-prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd churn-prediction
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### 4. Run the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the Customer Churn Prediction notebook and run the cells in order.

## 📂 Project Structure

```text
churn-prediction/
│
├── Customer_Churn_Prediction.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
```

*Make sure these filenames match the files you upload.*

## 🔮 Future Improvements

* Improve prediction performance through feature engineering.
* Compare the custom model with Scikit-learn's Logistic Regression.
* Evaluate precision, recall, and F1-score.
* Develop an interactive web interface.
* Deploy the model for real-time predictions.

## 👨‍💻 Author

**Rajat Tyagi**

B.Tech CSE (AI & ML) Student
IILM University, Greater Noida

<p>
  <a href="https://github.com/tyagirajat154">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://www.linkedin.com/in/rajattyagatt/">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
</p>

---

<div align="center">
  <i>Learning, building, and improving one project at a time.</i>
</div>
