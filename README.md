---

# 📧 **Marketing Email Campaign Optimization** 📈

Welcome to my project! 🎉 In this repo, I’ve analyzed an **email marketing campaign** to optimize the **click-through rate (CTR)** and gain insights into user behavior. 🚀

By applying models like **Logistic Regression**, **Gradient Boosting Classifier**, and **SVC**, we were able to identify patterns in email engagement and determine the most effective strategies to improve future campaigns. 📧🔍

---

## 📝 **Table of Contents**
1. [Project Overview](#project-overview)
2. [Data Files](#data-files)
3. [How to Run the Analysis](#how-to-run-the-analysis)
4. [Models Used](#models-used)
5. [Visualizations](#visualizations)
6. [Conclusion](#conclusion)
7. [Repository Structure](#repository-structure)

---

## 📊 **Project Overview**

In this project, the goal is to:
- **Analyze** an email marketing campaign to evaluate the **click-through rate** (CTR).
- **Optimize** email strategies by identifying key features that influence user engagement (e.g., email version, timing, user behavior).
- **Build models** to predict which factors are most important in increasing CTR.

We’ve used **Logistic Regression**, **Gradient Boosting**, and **SVC** (Support Vector Classifier) to model the data and determine which email features impact the success of campaigns.

---

## 💾 **Data Files**

This project uses three key datasets:

1. **email_table.csv**: Information on emails sent (e.g., ID, time sent, email version).
2. **email_opened_table.csv**: Data on which emails were opened.
3. **link_clicked_table.csv**: Data on which emails had the link clicked.

You can upload these files to your `data/` folder to run the analysis.

---

## 🛠️ **How to Run the Analysis**

1. **Clone the Repo:**

```bash
git clone https://github.com/Charish53/MarketingEmailCampaign.git
cd MarketingEmailCampaign
```

2. **Install Required Libraries:**

```bash
pip install -r requirements.txt
```

3. **Run the Jupyter Notebook:**

Open the Jupyter notebook `email.ipynb` and execute the cells to load data, preprocess it, train models, and visualize the results.

---

## 🧠 **Models Used**

In this project, I implemented the following classifiers to predict email link clicks:

### 1. **Logistic Regression** 📈
Logistic Regression is used as the baseline model. It predicts the probability of a user clicking the email link based on features like `email_version`, `user_country`, `hour_sent`, and more.

```python
from sklearn.linear_model import LogisticRegression
lr_params = {'C':[0.1,0.2, 0.3, 0.4]}
lr_clf = GridSearchCV(LogisticRegression(), lr_params, scoring='roc_auc')
lr_clf.fit(X_sub, y_sub)
```

### 2. **Gradient Boosting Classifier** 🌱
A powerful ensemble method that builds a series of weak learners to predict the target variable. It’s a great option when you need higher accuracy for imbalanced datasets.

```python
from sklearn.ensemble import GradientBoostingClassifier
gbc_params = {
    'learning_rate':[0.1,0.2,0.3,0.4],
    'n_estimators':[100,200],
    'min_samples_leaf':[1,5,7,8],
    'max_features':['auto','log2']
}
gbc_clf = GridSearchCV(GradientBoostingClassifier(), gbc_params, scoring='roc_auc')
gbc_clf.fit(X_sub, y_sub)
```

### 3. **Support Vector Classifier (SVC)** 🧑‍💻
SVC is used for classification tasks and can model complex non-linear relationships in the data by using different kernels like `rbf`, `sigmoid`, and `poly`.

```python
from sklearn.svm import SVC
svc_params = {
    'kernel':['rbf','sigmoid','linear','poly'],
    'C':[0.1,0.5,1,5,10],
    'degree':[1,3,5],
    'coef0':[0.5,0.6,0.7]
}
svc_clf = GridSearchCV(SVC(), svc_params, scoring='roc_auc')
svc_clf.fit(X_sub, y_sub)
```

---

## 📊 **Visualizations**

We generated multiple plots to visualize the performance of the models and evaluate their results. These visualizations include:

### 1. **ROC Curves** 🧩
The **Receiver Operating Characteristic** (ROC) curve shows the performance of each model by comparing the true positive rate to the false positive rate.

![ROC Curve](https://github.com/Charish53/MarketingEmailCampaign/blob/master/images/ROC.png)

### 2. **Confusion Matrix** 🔄
A confusion matrix shows how well our classifier performs in terms of false positives, false negatives, true positives, and true negatives.

![Confusion Matrix](https://github.com/Charish53/MarketingEmailCampaign/blob/master/images/confusion.png)

### 3. **Precision-Recall Curve** 💥
This curve helps evaluate the precision and recall for each class, which is especially useful for imbalanced datasets.

![Precision-Recall Curve](https://github.com/Charish53/MarketingEmailCampaign/blob/master/images/precision.png)

### 4. **Random Forest & Gradient Boosting Classifier Comparison 🔥**
We compare the performance of **Random Forest** and **Gradient Boosting** on the email data, observing the models' accuracy and precision.

![Random Forest vs Gradient Boosting](https://github.com/Charish53/MarketingEmailCampaign/blob/master/images/rf.png)

### 5. **Recall vs Threshold Plot** 🎯
A recall vs threshold plot shows the trade-off between recall and the classification threshold.

![Recall vs Threshold](https://github.com/Charish53/MarketingEmailCampaign/blob/master/images/re.png)

---

## 📈 **Conclusion**

By training and evaluating models like **Logistic Regression**, **Gradient Boosting**, and **SVC**, we identified key features that influence **click-through rates (CTR)**. The most impactful factors include **email personalization**, **timing**, and **user behavior** (past purchases).

- **Logistic Regression** performed well as a baseline model, but **Gradient Boosting** outperformed it in terms of accuracy.
- **Support Vector Classifier** also provided competitive results, particularly with its non-linear kernel transformations.
- With these findings, future campaigns can be optimized by targeting users who are most likely to engage based on their characteristics.

---

## 📂 **Repository Structure**

```plaintext
MarketingEmailCampaign/
├── data/
│   ├── email_table.csv
│   ├── email_opened_table.csv
│   ├── link_clicked_table.csv
├── notebooks/
│   ├── email.ipynb
├── images/
│   ├── ROC.png
│   ├── confusion.png
│   ├── precision.png
│   ├── ppd.png
│   ├── re.png
│   ├── rf.png
├── requirements.txt
└── README.md
```

---

## 📜 **requirements.txt**

This file contains all the dependencies required for this project:

```plaintext
pandas
numpy
seaborn
matplotlib
scikit-learn
imbalanced-learn
statsmodels
scipy
```

---

## 💬 **How to Contribute**

Feel free to fork this project, submit issues, or open pull requests if you find any improvements or bugs. Contributions are welcome! 🚀

---

## 📣 **Final Thoughts**

This project showcases how to analyze, optimize, and predict user engagement in email campaigns using machine learning algorithms. I hope you find it helpful and fun to explore! 🎉 If you have any questions, don't hesitate to reach out.

---

### **GitHub Repository Structure**

```plaintext
MarketingEmailCampaign/
├── data/
│   ├── email_table.csv
│   ├── email_opened_table.csv
│   ├── link_clicked_table.csv
├── notebooks/
│   ├── email.ipynb
├── images/
│   ├── ROC.png
│   ├── confusion.png
│   ├── precision.png
│   ├── ppd.png
│   ├── re.png
│   ├── rf.png
├── requirements.txt
└── README.md
```

---

### **Final Notes on Images**:

- Make sure to upload the images (`ROC.png`, `confusion.png`, `precision.png`, `ppd.png`, `re.png`, `rf.png`) to the **`images/`** folder in the repository.
- Replace `https://github.com/Charish53/MarketingEmailCampaign/blob/master/images/xyz.png` with the correct links to your images once they are uploaded.

---

This **README.md** is structured with engaging emojis and clear explanations of the models used and results obtained. It provides a professional yet fun look for your repository. Let me know if you need any more changes! 😊
