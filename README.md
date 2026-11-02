# Customer Churn Prediction

An end-to-end machine learning web app that predicts whether a telecom customer is likely to leave. It includes a logistic regression built **from scratch in NumPy**, benchmarked against scikit-learn's, and served through a Flask API with a five-page dashboard.

**Live demo:** https://churn-prediction-40tz.onrender.com

> The app runs on a free hosting plan and sleeps when idle, so the first load can take 30-60 seconds.

---

## Screenshots

### 1. Churn Prediction
Enter a customer's details and get a probability, a risk band, the factors behind the score, and a what-if simulation.

<img src="images/prediction.png" width="100%" alt="Churn Prediction page">

### 2. Dashboard
Dataset overview and model summary, read from numbers computed in the notebook.

<img src="images/dashboard.png" width="100%" alt="Dashboard page">

### 3. Customer Insights
Observed churn rates by contract, tenure, payment method and internet service.

<img src="images/insights.png" width="100%" alt="Customer Insights page">

### 4. Model Performance
Test-set metrics, ROC curve and confusion matrix for both models, plus a train vs test check.

<img src="images/performance.png" width="100%" alt="Model Performance page">

### 5. Settings
A read-only description of the data, preprocessing, models, limitations and privacy notes.

<img src="images/settings.png" width="100%" alt="Settings page">

---

## Features

- **Churn prediction form:** 19 customer fields in, a churn probability, a predicted class and a risk band (Low / Medium / High) out.
- **Two models side by side:** the custom NumPy model and the scikit-learn model are both scored on every prediction.
- **Explanation:** each feature's contribution to the score, compared with an average training customer.
- **What-if simulation:** change contract, payment method, tech support or online security and the model re-scores the customer.
- **Input validation:** invalid values return a clear error instead of a wrong prediction.
- **Dashboard, Insights, Model Performance and Settings pages**, all filled from `artifacts/stats.json`.

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 customers, 21 columns.

- 11 rows were dropped because `TotalCharges` was blank, leaving **7,032** customers.
- **1,869** customers churned (**26.58%**).
- `customerID` is removed before training.

---

## How the model works

### The problem
This is **binary classification**: for each customer, predict `Churn = 1` (leaves) or `Churn = 0` (stays). The model outputs a probability between 0 and 1, and a threshold of 0.50 turns it into a yes/no decision.

### Logistic regression in three steps

1. **Weighted sum.** Each customer is a row of 30 numbers (tenure, charges, one-hot columns such as `Contract_Two year`). The model multiplies each by a learned weight and adds a bias:

   ```
   z = w1*x1 + w2*x2 + ... + w30*x30 + b
   ```

2. **Sigmoid.** The sum `z` can be any number, so the sigmoid squeezes it into a probability:

   ```
   p = 1 / (1 + e^(-z))
   ```

3. **Decision.** If `p >= 0.50` the customer is predicted to churn.

A positive weight pushes the probability up, a negative weight pushes it down.

### How the custom model learns (gradient descent)

Weights and bias start at zero. Then, 500 times:

1. Predict probabilities for all training customers.
2. Compute the error: `error = prediction - actual`.
3. Compute the gradients: `dw = X^T * error / n` and `db = mean(error)`.
4. Update: `weights -= learning_rate * dw` and `bias -= learning_rate * db`.

The quantity being minimised is the **binary cross-entropy (log-loss)**. In the notebook it fell steadily and flattened out, which shows the training converged:

| Iteration | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|
| Loss | 0.4303 | 0.4186 | 0.4160 | 0.4150 | 0.4145 |

Settings: learning rate 0.1, 500 iterations, no regularization.

### scikit-learn model
`LogisticRegression(max_iter=1000, random_state=42)` solves the same problem with a more advanced optimiser and applies L2 regularization by default. The custom model landing within about 1.3 percentage points of it on every metric is evidence that the from-scratch implementation is correct. The small gap is likely due to the different optimiser and regularization, and the custom model's loss was still dropping slightly at 500 iterations.

### Why preprocessing matters
- **Split before encoding.** The 80/20 split (stratified) happens first, so nothing from the test set influences training.
- **Scaling.** `StandardScaler` is fitted on the training data only, then applied to the test data. Scaling matters for gradient descent, because features on very different scales make learning slow and unstable.
- **Same pipeline at prediction time.** The app loads the saved scaler and the exact list of 30 columns (`feature_columns.json`), so a customer entered in the form goes through the same transformations as the training data.

### How the explanation is computed
For logistic regression the contribution of each feature is exact:

```
contribution = weight x scaled value
```

A scaled value of 0 means "average training customer", so each bar on the prediction page is how far that feature moves the score (in log-odds) compared with an average customer. Positive pushes toward churn, negative pushes away. Tenure, monthly charges and total charges are strongly correlated, so read those three together rather than one at a time. This is not SHAP; for a linear model it is the direct calculation.

### Risk bands
Low below 30%, Medium 30-60%, High 60% and above. These are a hand-chosen labelling convention, not learned by the model.

---

## Methodology summary

1. Convert `TotalCharges` to numeric, drop blank rows, encode `Churn` as 1/0.
2. Stratified 80/20 split: 5,625 train / 1,407 test, `random_state=42`.
3. One-hot encode with `drop_first=True`, align test columns to training columns (30 features).
4. `StandardScaler` fitted on training data only.
5. Train both models on the same split and evaluate on the same test set.
6. Save the models, scaler, column order and dashboard statistics to `artifacts/`.

## Results (test set, threshold 0.50)

| Metric | Custom (NumPy) | scikit-learn |
|---|---|---|
| Accuracy | 0.7982 | 0.8038 |
| Precision | 0.6347 | 0.6476 |
| Recall | 0.5668 | 0.5749 |
| F1-score | 0.5989 | 0.6091 |
| ROC-AUC | 0.8325 | 0.8357 |

Confusion matrices (1,407 test customers):

| | TN | FP | FN | TP |
|---|---|---|---|---|
| Custom | 911 | 122 | 162 | 212 |
| scikit-learn | 916 | 117 | 159 | 215 |

**Reading the results**
- **ROC-AUC about 0.83** means the model ranks customers well: a randomly chosen churner gets a higher score than a randomly chosen non-churner about 83% of the time.
- **Accuracy of about 80% should be read carefully.** 73% of customers do not churn, so predicting "stay" for everyone would already score about 73%.
- **Recall of about 0.57** is the weak point: roughly 4 in 10 churners are missed at the 0.50 threshold.
- **No overfitting:** train and test scores are close (scikit-learn accuracy 0.8043 train vs 0.8038 test).

### Patterns in the data (observed churn rates)

- Contract: month-to-month 42.7%, one year 11.3%, two year 2.9%.
- Tenure: 0-12 months 47.7%, 49-72 months 9.5%.
- Payment method: electronic check 45.3%, credit card (automatic) 15.3%.

These are correlations, not proof of cause.

## Limitations

- Recall is about 0.57. Threshold tuning has not been done yet.
- A single train-test split was used. There is no cross-validation, so the metrics have no error bars.
- Probability calibration has not been checked.
- Fairness across customer groups has not been evaluated.
- The risk bands are a labelling convention, not learned.
- Trained on one public dataset. It may not transfer to another company or period.
- The app has no login. This is a student project, not a production system. Do not enter real customer data.

## Tech stack

Python 3.12, NumPy, pandas, scikit-learn, Flask, gunicorn, Chart.js, HTML/CSS/JavaScript. Hosted on Render. UI designed with Google Stitch.

## Project structure

```
.
├── app.py                  # Flask app: validation, preprocessing, prediction API, page routes
├── Churn_project.ipynb     # Data cleaning, both models, evaluation, saves artifacts/
├── requirements.txt
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── artifacts/
│   ├── sklearn_model.joblib
│   ├── custom_model.npz
│   ├── scaler.joblib
│   ├── feature_columns.json
│   └── stats.json          # numbers shown on the dashboard pages
├── static/                 # index, dashboard, insights, performance, settings (HTML)
└── images/                 # screenshots used in this README
```

## Run locally

```bash
git clone https://github.com/tyagirajat154/churn-prediction.git
cd churn-prediction
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

Use Python 3.12. The saved models were created with the package versions in `requirements.txt`, so other versions may fail to load them. To retrain, run `Churn_project.ipynb` from top to bottom; it regenerates everything in `artifacts/`.

## API

`POST /api/predict` with a JSON body of the 19 customer fields (and an optional `"model": "sklearn"` or `"custom"`). It returns the churn probability, predicted class, risk band, both models' probabilities and the top contributing factors. `GET /api/health` returns the service status.

## Future work

- Tune the decision threshold to improve recall.
- Cross-validation and a calibration check.
- Try other algorithms and compare them fairly.
- Add authentication before any use with real data.

## Author

**Rajat Tyagi**, B.Tech CSE (AI & ML), IILM University, Greater Noida
[GitHub](https://github.com/tyagirajat154)
