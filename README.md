### Capstone Project: Student Depression Prediction

**Author: Chiayu Peng**

#### Executive Summary

This project develops a supervised classification machine learning model to predict depression in studnets using data such as demographics, academic indicators, lifestyle, and other factors. The goal is to minimize missed depression cases (false negatives) by optimizing for recall score, ensuring at-risk students are identified and referred to mental wellnes institutes or practioners for early interventions, while identifying the top 5 drivers impacting a student's emotional wellbeing. 

A KNN model is chosen as the final model for deployment because it achieves the highest recall score (0.9227).

#### Rationale
It is important to be able to identify students who may require intervention for their emotional wellbeing. If the question is left unanswered, students who are emotionally unwell may suffer in silence without anyone ever noticing it. With a predictive model capable of identifying potentially emotionally unwell students, emotional and mental wellness institutions can gain insight on factors contributing to depress among students, and the predictive model can flag potentially depressed students to introduce early interventions.

#### Research Question
What are the top factors contributing to a student's mental wellness?

#### Data Source
The data is sourced from the Kaggle [website](https://www.kaggle.com/datasets/adilshamim8/student-depress-dataset/data), titled "Student depress Dataset". The dataset has 18 columns and total of 27901 rows, which is adequate for data analysis. Based on the Kaggle website, the data was compiled through multiple datasets from OpenML.org, and senstitive information was removed or encoded to achieve anonymity. The dataset is adequate for reasearch purposes.

**Feature Description**

- **id**: A unique identifier assigned to each student record in the dataset..
- **Gender**: The gender of the student, Male or Female.
- **Age**: The age of the studnet in years.
- **City**: The city or region where the student resides, providing geographical context for the analysis.
- **Profession**: The field of work or study of the student, which may offer insights into occupational or academic stress factors.
- **Academic Pressure**: A measure indicating the level of pressure the student faces in academic settings. This could include stress from exams, assignments, and overall academic expectations. 0-5.
- **Work Pressure**: A measure of the pressure related to work or job responsibilities, relevant for students who are employed alongside their studies. 0-5.
- **CGPA**: The cumulative grade point average of the student, reflecting overall academic performance. 0-10.
- **Study Satisfaction**: An indicator of how satisfied the student is with their studies, which can correlate with mental well-being. 0-5.
- **Job Satisfaction**: A measure of the student’s satisfaction with their job or work environment, if applicable.
- **Sleep Duration**: The average number of hours the student sleeps per day, which is an important factor in mental health. 0-4. Less than 5 hours, 7-8 hours, and Others.
- **Dietary Habits**: An assessment of the student’s eating patterns and nutritional habits, potentially impacting overall health and mood. Healty, Unhealthy, Moderate, Others.
- **Degree**: The academic degree or program that the student is pursuing. Various.
- **Have you ever had suicidal thoughts ?**: A binary indicator (Yes/No) that reflects whether the student has ever experienced suicidal ideation. 
- **Work/Study Hours**: The average number of hours per day the student dedicates to work or study, which can influence stress levels. 0-12.
- **Financial Stress**: A measure of the stress experienced due to financial concerns, which may affect mental health. 1-5.
- **Family History of Mental Illness**: Indicates whether there is a family history of mental illness (Yes/No), which can be a significant factor in mental health predispositions.
- **Depression**: The target variable that indicates whether the student is experiencing depress (Yes/No = 0/1). This is the primary focus of the analysis. 

#### Methodology

1. Exploratory Data Analysis
2. Data Cleaning and Preprocessing
3. Feature Engineering
4. Data Splitting
5. Model Selection - KNN, Logistic Regression, Random Forest, SVM, and XGBoost. Baseline: Simple Logistic Reression
6. Model Training
7. Hyperparameter Tuning
8. Model Evaluation - Recall Score
9. Model Interpreation and Explainability - SHAP Values
10. Model Validation and Testing

#### Results from Best Performing Models

| Model              | Train Recall | Val Recall | Train→Val Gap | Status       |
|--------------------|--------------|------------|---------------|--------------|
| KNN                | 0.9230       | 0.9227     | 0.03%         | Generalizing |
| Logistic Regression| 0.9074       | 0.9083     | -0.09%        | Generalizing |
| XGBoost            | 0.8987       | 0.9009     | -0.22%        | Generalizing |
| SVM                | 0.9017       | 0.8923     | 0.93%         | Generalizing |
| Random ForestSVM   | 0.9009       | 0.8907     | 0.93%         | Generalizing | 

- **Final model for deployment**: KNN (n_neighbors=178, uniform weights, euclidean metric)
- **Validation recall**: 0.9227 — 92.3% of depressed patients correctly identified
- **Generalization gap**: 0.3% — model generalizes well to unseen data

#### Business Impact

- **92.3% of depressed students** are correctly identified
- **False negative rate of 7.8%** — 1 in 13 depressed students missed
- Compared to baseline (Logistic Regression recall = 0.888):
  - KNN catches **3.47% more** depressed patients
  - At 27,000 patients this means ~936 additional cases identified

#### Limitations

- KNN is computationally expensive at inference time on large datasets
- Model explainability is approximate (KernelExplainer) — less precise than tree-based SHAP
- Model requires retraining if student demographics shift significantly
- Model is biased towards students <= 25 years old (recall = 0.96) compared to students (recall = 0.89)
- Model is trained using data collected in India and may fail to predict depression in students in other countries


#### Model Monitoring and Maintenance


#### Future Improvement and Next Steps

**Data Collection**
- Proper data collection on student degree, to highlight which majors may be more likley to contribute to student depression
- Names of city alone are not useful. Consider appending features describing living conditions for the city, such as population density, amount of green spaces, air pollution index, crime rate, infratructure / transport, healthcare access, etc. 
- Include lifetyle features beyond sleep hours and dietary habits, such as exercise hours, socialization / leisure hours, screen time hours.
- Instead of binary values (1/0) for `Depressoin`, consider using a scale to focus on those who are the most depressed
- The dataset was collected within India. Consider a global survey to expand samples beyond students in India

#### Outline of Project

- [Link to notebook](https://github.com/chiayu-peng/capstone/blob/main/capstone_chiayupeng.ipynb)

**Repository Structure**

├── data/
│   ├── student_depression_dataset.csv   # original dataset
├── capstone_chiayupeng.ipynb            # jupyter notebook
└── README.md

##### Contact and Further Information
Email: cpeng26@gmail.com
