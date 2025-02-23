# Loan Default Prediction Model

## Overview

This project develops a machine learning model to predict loan defaults using historical loan data from Lending Club. The primary goal is to build a scorecard that assigns risk scores to loan applicants, aiding in the assessment of their creditworthiness. The project encompasses data preprocessing, feature engineering, model selection, and out-of-time validation to ensure robust and generalizable performance.

## Data

The dataset used in this project is obtained from Lending Club. It includes information about accepted loan applications, such as loan amount, interest rate, borrower demographics, and loan status.

*   `01_accepted_2007_to_2018Q4.csv`: Contains accepted loan applications data from 2007 to Q4 2018.
    *   A subset of this data, specifically loans issued in August-December 2014, is used for training.
    *   Loans issued in Jan-Mar 2015 are used for Out-of-Time validation

## Key Steps

1.  **Data Loading and Preparation:**
    *   Loads loan data from CSV files.
    *   Creates a binary target variable (`Bad_Flag`) indicating loan default.
    *   Subsets the data for training and out-of-time validation.

2.  **Exploratory Data Analysis (EDA) and Feature Engineering:**
    *   Analyzes categorical and date variables.
    *   Drops irrelevant columns.
    *   Creates the target variable 'Bad_Flag' based on 'loan_status'.

3.  **Missing Value Imputation:**
    *   Identifies and handles missing values using a tiered approach:
        *   Drops variables with >= 80% missing values.
        *   Imputes variables with 50-80% missing values with 99999 (or 0 in OOT validation due to data differences).
        *   Imputes variables with < 50% missing values using a bad-rate informed median imputation.

4.  **Outlier Handling:**
    *   Caps outliers in numerical variables at the 5th and 95th percentiles.

5.  **One-Hot Encoding:**
    *   Converts categorical features into numerical features using one-hot encoding.

6.  **Feature Selection:**
    *   Calculates KS statistic and Gini coefficient for each numerical variable.
    *   Performs Pearson correlation analysis.
    *   Manually selects a subset of variables based on statistical significance and business intuition.
    *   Calculates Variance Inflation Factor (VIF) to check for multicollinearity.

7.  **Model Training and Evaluation:**
    *   Explores three classification algorithms: Ridge Regression, Random Forest, and Gradient Boosting Machine (GBM).
    *   Splits data into training and testing sets.
    *   Trains each model on the training data.
    *   Evaluates models using accuracy and log loss.

8.  **Out-of-Time (OOT) Validation:**
    *   Tests the model on a separate OOT dataset (January-March 2015 loan applications).
    *   Calculates accuracy and log loss on the OOT data.

9.  **Scorecard Development:**
    *   Develops a scorecard based on the predicted probabilities from the Random Forest model.
    *   Scales probabilities to create a score range from 300 to 850.

## Code Structure

The project code is organized as follows:

*   **Import Libraries:** Imports necessary libraries, including NumPy, Pandas, scikit-learn, statsmodels, matplotlib, and seaborn.
*   **Data Loading and Preparation:** Reads and preprocesses the loan data.
*   **EDA and Feature Engineering:** Explores the data and engineers relevant features.
*   **Missing Value Imputation:** Handles missing values using the described methods.
*   **Outlier Handling:** Caps outliers to reduce their impact.
*   **One-Hot Encoding:** Converts categorical variables to numerical format.
*   **Feature Selection:** Selects relevant features using statistical and business criteria.
*   **Model Training and Evaluation:** Trains and evaluates Ridge Regression, Random Forest, and GBM models.
*   **OOT Validation:** Validates the model on out-of-time data.
*   **Scorecard Development:** Creates a risk scorecard based on model predictions.

## Libraries Used

*   `numpy`: Numerical computing
*   `pandas`: Data manipulation and analysis
*   `matplotlib`: Data visualization
*   `seaborn`: Statistical data visualization
*   `scikit-learn`: Machine learning algorithms and tools
*   `statsmodels`: Statistical modeling and econometrics
*   `scipy`: Scientific computing

## Results

*   **Model Performance:**  Random Forest achieved the best performance with a test accuracy of 98.41% and an OOT accuracy of 98.36%.
*   **Feature Importance:**  Key predictors of loan default include `recoveries`, `last_fico_range_high`, `total_rec_prncp`, `last_pymnt_amnt`, and `int_rate`.
*   **Scorecard:**  The developed scorecard assigns risk scores from 300 to 850, providing a practical tool for assessing loan applicant risk.

| Algo          | Train_Error         | Train_Acc | OOT_Error           | OOT_Acc |
| :------------ | :------------------ | :-------- | :------------------ | :------ |
| Ridge         | 0.0822002245989056  | 0.9662    | 0.08556710142894697 | 0.9641  |
| GBM           | 0.07316159106628008 | 0.9753    | 0.07599692128878723 | 0.9741  |
| Random Forest | 0.05165800250375114 | 0.9841    | 0.06180492322965778 | 0.9836  |

## Usage

1.  **Clone the repository:**
    ```
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install the required libraries:**
    ```
    pip install numpy pandas matplotlib seaborn scikit-learn statsmodels scipy
    ```

3.  **Run the code:**
    ```
    jupyter notebook loan-default-prediction.ipynb
    ```

## Next Steps

*   Further refine feature selection to improve model interpretability and reduce complexity.
*   Explore additional machine learning algorithms and hyperparameter tuning techniques.
*   Implement the scorecard in a production environment for real-time loan risk assessment.
*   Incorporate external data sources to enhance the predictive power of the model.

