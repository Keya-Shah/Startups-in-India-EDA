1. **Initial Analysis:**
   -First we performed basic data analysis , which  included checking the shape, examining data types, looking for missing values, checking the number of unique values in each column, identifying duplicate rows and knowing the mathematical description of data.

2. **Exploratory Data Analysis (EDA) and Preprocessing:**
   - Then we conducted EDA is to gain insights into the dataset. This includes visualizations and analyses of variables like blood pressure, body measurements, haemoglobin levels, cholesterol levels, etc.
   - Gender-wise analysis was done. We also compared different factors for the smoking and drinking groups through boxplots and histplots.
   - We encoded categorical variables (e.g., 'sex' and 'DRK_YN') using label encoding. Also, removed duplicate rows from the dataset.

3. **Classification Problems:**
   - We identified two classification tasks:
     - Task 1: Predicting Smoking Status (SMK_stat_type_cd)
     - Task 2: Predicting Drinking Status (DRK_YN)

4. **Model Building and Evaluation:**
   - For each classification task:
     - We first used all the features and trained and evaluated different models (Decision Tree, Logistic Regression and Support Vector Classifier)
     - Then selected Features based on correlation with the target variable. Trained and evaluated different models (Decision Tree, Random Forest, Logistic Regression, and Support Vector Classifier) 
     - Performed hyperparameter tuning for random forest and logistic regression models. 
     - Classification reports were generated to assess model performance.
