# Part 1 – Churn Prediction with Python

## Overview
This project addresses the critical challenge of customer churn within a bank’s credit card services. Leveraging Python, it encompasses comprehensive data exploration, strategic feature engineering, and robust predictive modeling. The primary objective is to equip the bank with actionable insights, accurately identifying customers most likely to churn to enable timely intervention and effective retention strategies.

## Dataset
* **Source:** [Kaggle – Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/data)
* **Records:** Approximately 10,000 customer profiles, encompassing demographic information, credit card usage patterns, and churn status.
* **Target Variable:** `Attrition_Flag` (binary: 'Existing Customer' vs. 'Attrited Customer').

## Key Steps
1.  **Introduction & Business Context**
2.  **Dataset Overview & Initial Observations**
3.  **Data Cleaning & Exploratory Data Analysis (EDA)**
4.  **Feature Engineering**
5.  **Model Development & Evaluation**
6.  **Model Insights & Deployment**

## Tech Stack
The project utilizes the following Python libraries and environment:

* **Python Libraries:**
    * `pandas`: For data manipulation and analysis.
    * `numpy`: For numerical operations.
    * `matplotlib.pyplot`: For creating static visualizations.
    * `seaborn`: For enhanced statistical data visualization.
    * `scikit-learn`: For machine learning tasks including model selection, preprocessing, and various classification algorithms.
* **Environment:** Jupyter Notebook

## Introduction & Business Context
### Business Problem
The bank is experiencing a growing concern: an increasing number of customers are closing their credit card accounts. This churn trend poses significant financial and strategic challenges. Leadership seeks to understand which customers are likely to attrite, enabling proactive intervention with tailored services or offers to mitigate losses.

### Project Objective
* To explore customer behavior and account data to uncover patterns related to churn.
* To engineer relevant features from raw variables to enhance predictive power.
* To build and evaluate robust predictive models capable of estimating churn risk.
* To provide actionable insights that inform targeted customer retention strategies.

## Dataset Overview & Initial Observations
### Source
This dataset is publicly available on [Kaggle: Credit Card Customers Dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/data). It contains detailed information on 10,127 customers, including demographic attributes, account activity, and churn status.

### Features
The dataset comprises 18 original variables spanning customer profiles and credit card usage patterns.

### Initial Observations
The dataset was found to be generally clean and well-structured, with minimal missing values. Features include a mix of categorical and numerical types, making them suitable for both statistical analysis and machine learning model development.

## Data Cleaning & Exploratory Data Analysis (EDA)
In this step, raw data was loaded, unnecessary columns were removed, and column names were updated for improved readability and ease of use. Categorical features were strategically encoded:

* **Ordinal Encoding:** Applied to variables with inherent order, such as education level and income level, to represent their natural hierarchy numerically.
* **One-Hot Encoding:** Applied to nominal categorical variables like gender, marital status, and card_type to convert them into a suitable numerical format for machine learning models without implying order.

## Feature Engineering
This phase involved introducing a set of new, highly informative features designed to capture more nuanced customer behaviors and credit dynamics, thereby enhancing the model's predictive capabilities. Engineered features include:

* **Behavioral Metrics:** Such as transaction amount per month, transactions per month, and average transaction amount.
* **Credit Dynamic Metrics:** Including balance to limit ratio, credit usage efficiency, and credit growth rate.

## Model Development & Evaluation
### Preparing Data for Modeling
The dataset was meticulously prepared for machine learning by separating the target variable (`churn_flag`) and `client_id` from the features. The dataset was then split into training and testing chunks to ensure unbiased model evaluation. Numerical columns underwent standardization using `StandardScaler` to bring them to a common scale, which is crucial for many machine learning algorithms.

### Training Classification Models
Three distinct classification models were trained and evaluated:

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier

### Model Evaluation
The performance of each trained model was comprehensively evaluated using a suite of metrics, including Accuracy, Precision, Recall, F1-Score, and ROC AUC. Visualizations such as ROC Curves and Confusion Matrices were utilized to provide deeper insights into model behavior and to identify the model demonstrating the greatest predictive power and generalizability.

![ROC Curve Gradient Boosting](/Images/roc_curve_gradient_boosting.png "ROC Curve Gradient Boosting")
![ROC Curve Logistic Regression](/Images/roc_curve_logistic_reg.png "ROC Curve Logistic Regression")
![ROC Curve Random Forest](/Images/roc_curve_random_forest.png "ROC Curve Random Forest")

## Model Insights & Deployment
### Key Results & Model Selection
Upon comprehensive evaluation, the Gradient Boosting Classifier emerged as the superior model. It achieved the highest overall performance across all key metrics:

* **Accuracy:** 0.9605
* **Precision:** 0.9398
* **Recall:** 0.8059
* **F1-Score:** 0.8677
* **ROC AUC:** 0.9887

### Feature Importance
A detailed feature importance analysis was conducted on the chosen Gradient Boosting model to identify the most influential variables. This analysis revealed `trans_ct` (transaction count) as overwhelmingly the most important feature, significantly contributing to the model's predictive decisions. Other key features included `revolving_bal`, `trans_amt`, and `ct_chg_q4_q1`.
![Feature Importance Gradient Boosting](/Images/feature_importance_gradient_boost.png "Feature Importance Gradient Boosting")


### Optimal Categorization Threshold Identification
To translate model probabilities into actionable classifications, the optimal threshold for predicting churn was identified using the precision-recall curve. Analysis indicated a "sweet spot" for the threshold somewhere between **0.3 and 0.4**, offering a balanced trade-off between precision and recall, crucial for effective intervention strategies.
![Precision Recall Curve Gradient Boosting](/Images/precision_recall_curve.png "Precision Recall Curve Gradient Boosting")


### Model Deployment & Actionable Data
In the final step, the trained Gradient Boosting model was deployed. It was applied to the entire original dataset to generate churn probabilities for every customer. Utilizing the optimized threshold of **0.38**, customers were classified as 'predicted churn' or 'not churn'. Crucially, **126 customers (representing 1.24% of the base)** were identified as "currently at-risk" (predicted to churn but not yet attrited), providing a highly actionable segment for targeted retention efforts. The resulting dataset was then prepared and cleaned for seamless import into Power BI, enabling interactive dashboards for ongoing monitoring and strategic decision-making.

---

🎯 This notebook serves as the analytical engine behind the full 3-part project. The robust model and key insights generated here will directly power interactive dashboards in Part 2 and guide strategic decision-making and intervention strategies in Part 3.