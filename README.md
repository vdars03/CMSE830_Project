
# Student Performance Explorer

An interactive Streamlit dashboard examining associations between student characteristics, study habits, academic history, and final mathematics grades.

Built as a CMSE 830 data science project at Michigan State University.

## 1. Project Overview

Student Performance Explorer helps users understand how student characteristics, study habits, attendance, previous academic performance, and educational support relate to final mathematics grades.

The application combines exploratory data analysis, interactive visualizations, statistical correlations, and predictive modeling to investigate student performance.

The goal is to communicate meaningful patterns in educational data while recognizing that correlations do not establish causation.

## 2. Target Audience

The dashboard is designed for:

- Students interested in understanding academic performance patterns.
- Educators exploring relationships between study habits and grades.
- Academic advisors interested in factors associated with student outcomes.
- Data science learners exploring statistical analysis and predictive modeling.

The application supports exploratory understanding rather than individual student evaluation or high-stakes academic decisions.

## 3. Research Questions

The project investigates the following questions:

1. How does weekly study time relate to final mathematics grades?
2. What relationship exists between student absences and academic performance?
3. How do previous class failures relate to final grades?
4. How does family or school educational support relate to student performance?
5. Which numerical variables have the strongest associations with final grades?
6. How much do previous grades improve the accuracy of final-grade predictions?

## 4. Dataset

**Source:** UCI Machine Learning Repository — Student Performance Dataset

Dataset URL:
https://archive.ics.uci.edu/dataset/320/student+performance

**Reference:** Cortez and Silva (2008)

**File:** `student-mat.csv`

**Dataset characteristics:**

- 395 student records
- 33 original variables
- Mathematics course performance
- Semicolon-delimited CSV file
- Final grade measured on a 0–20 scale

Each observation represents one student record.

### Important Variables

| Variable | Description |
|---|---|
| age | Student age |
| Medu | Mother's education level |
| Fedu | Father's education level |
| studytime | Weekly study-time category |
| failures | Number of previous class failures |
| absences | Number of recorded absences |
| famsup | Family educational support |
| schoolsup | Extra school educational support |
| G1 | First-period grade |
| G2 | Second-period grade |
| G3 | Final mathematics grade |

The target variable is `G3`.

`G1` and `G2` represent earlier grading periods and are not available at the beginning of the school year.

## 5. Application Features

The application contains five main sections.

### Home

Introduces the project, explains its purpose, identifies the target audience, and provides navigation instructions.

Displays:

- Total number of students
- Number of dataset variables
- Average final mathematics grade

### Explore Data

Provides interactive exploratory analysis through six tabs:

1. Overview
2. Study Time
3. Absences
4. Past Failures
5. Support
6. Correlations

Users can filter observations by:

- Maximum number of absences
- Weekly study-time categories

All exploratory charts and statistics update based on the selected students.

### Analysis & Modeling

Compares two linear regression models for predicting final mathematics grades.

Includes:

- Model predictor descriptions
- Held-out test evaluation
- MAE, RMSE, and R²
- Actual-versus-predicted scatterplots
- Residual distributions
- Five-fold cross-validation
- Model comparison charts

### Data Preparation

Documents:

- Missing-value assessment
- Duplicate-record checks
- Variable types
- Unique-value counts
- Feature engineering
- Predictor selection
- Preprocessing decisions

### Limitations & Documentation

Explains the dataset's limitations, modeling assumptions, responsible interpretation, and data source.

## 6. Data Preparation and Feature Engineering

The application examines data quality before performing analysis.

### Missing Values

The application checks missing values for every variable and reports missing-value percentages.

No imputation is applied when the dataset contains no missing values.

### Duplicate Records

Duplicate records are counted and reported.

Records are not automatically removed because identical attribute values do not necessarily represent the same student.

### Variable Types

The application displays each variable's data type and number of unique values.

This helps users understand which variables are numerical, ordinal, or categorical.

### Engineered Features

Three descriptive features are created.

**Absence groups:**

- 0 absences
- 1–5 absences
- 6–10 absences
- 11 or more absences

**Study-time groups:**

- Less than 2 hours per week
- 2–5 hours per week
- 5–10 hours per week
- More than 10 hours per week

**Final-grade groups:**

- Below 10
- 10–14
- 15–20

These engineered features support exploratory analysis and are not used as regression predictors.

### Scaling and Encoding

Ordinary least-squares linear regression does not require feature scaling to produce its fitted predictions.

The selected regression predictors are numerical or ordinal, so additional categorical encoding is not used.

## 7. Exploratory Data Analysis

The dashboard uses multiple visualization types selected to answer specific analytical questions.

### Final-Grade Distribution

**Visualization:** Histogram

**Purpose:** Examine how frequently different final mathematics grades occur.

A histogram is appropriate because it displays the frequency distribution of a numerical variable.

The dashboard also reports the most frequent grade and median grade for the selected students.

### Study Time and Final Grades

**Visualizations:**

- Average-grade bar chart
- Grade-distribution boxplot
- Group summary table
- Spearman correlation

Bar charts compare average grades across study-time categories.

Boxplots display medians, variability, and potential outliers within each category.

Spearman correlation measures the direction and strength of the rank association between study time and final grades.

### Absences and Final Grades

**Visualizations:**

- Grade-distribution boxplot
- Absence-group summary table
- Spearman correlation

The boxplot compares grade distributions across different absence ranges.

The correlation measures the rank association between recorded absences and final grades.

### Previous Failures and Final Grades

**Visualizations:**

- Average-grade bar chart
- Grade-distribution boxplot
- Spearman correlation

These plots examine how previous academic failures relate to final mathematics performance.

### Family and School Support

**Visualizations:**

- Average-grade bar charts
- Grade-distribution boxplots
- Group summary tables

These comparisons investigate differences in grades between students who report receiving educational support and those who do not.

Support may be provided to students who are already struggling, so observed differences cannot establish whether support causes changes in grades.

### Correlation Heatmap

**Visualization:** Spearman correlation heatmap

The heatmap compares rank associations among selected numerical and ordinal variables.

A diverging color scale centered at zero distinguishes positive and negative associations.

The application also displays a ranked table of correlations with the final grade.

This helps users identify which variables have the strongest observed associations with `G3`.

## 8. Evidence-Based Interpretation

The application includes automatically generated interpretations based on the current dataset selection.

### Group Comparisons

For selected group comparisons, the application:

1. Calculates average final grades.
2. Identifies the highest- and lowest-average groups with at least five students.
3. Reports the difference in average grades.
4. Explains that the differences are descriptive rather than causal.

### Correlation Interpretation

The application calculates Spearman correlation coefficients.

Spearman correlation ranges from -1 to +1.

- Positive values indicate a positive rank association.
- Negative values indicate a negative rank association.
- Values near zero indicate little or no monotonic rank association.

The strength and direction of the observed association are reported using the calculated coefficient.

Correlations do not establish causation.

### Filtered Analysis

Exploratory interpretations update when users modify the absence or study-time filters.

This allows users to examine how descriptive patterns differ across selected subsets.

Results from small groups should be interpreted cautiously.

## 9. Predictive Modeling

The project compares two ordinary least-squares linear regression models.

### Model 1 — Without Previous Grades

Predictors:

- age
- Medu
- Fedu
- studytime
- failures
- absences

Target:

`G3`

This model examines how selected student characteristics and academic-history variables predict final grades without using earlier grading-period results.

Recorded absences must be known at the intended prediction time.

### Model 2 — With Previous Grades

Predictors:

- age
- Medu
- Fedu
- studytime
- failures
- absences
- G1
- G2

Target:

`G3`

This model investigates whether earlier grading-period grades improve final-grade prediction accuracy.

Because `G1` and `G2` become available during the school year, this model is not suitable for beginning-of-year prediction.

### Train/Test Split

Both models use:

- 80% training data
- 20% testing data
- `random_state=42`

The same train/test observations are used for both models to support a fair comparison.

### Evaluation Metrics

**Mean Absolute Error (MAE)**

Measures the average absolute difference between predicted and actual grades.

Lower MAE indicates smaller average prediction errors.

**Root Mean Squared Error (RMSE)**

Measures prediction error while giving greater weight to larger errors.

Lower RMSE indicates better predictive accuracy.

**R-squared (R²)**

Measures the proportion of variation in final grades explained by the model relative to a constant-mean baseline.

Higher R² generally indicates better predictive performance on the evaluated data.

Negative R² values are possible.

### Model Comparison

The application displays both models' MAE, RMSE, and R² on the held-out test set.

It also automatically reports how the test MAE and R² change when previous grades are included.

This allows users to evaluate whether additional academic-history information improves prediction accuracy.

## 10. Model Validation

Five-fold cross-validation is performed on the training data.

The training observations are divided into five folds.

Each fold is used once for validation while the remaining folds are used for training.

The application reports:

- Mean cross-validation MAE
- Standard deviation of cross-validation MAE
- Mean cross-validation RMSE
- Mean cross-validation R²

A line chart displays validation MAE across the five folds for both models.

Cross-validation provides additional evidence about model stability across different training-data partitions.

The held-out test data are not used during cross-validation.

## 11. Model Diagnostic Visualizations

### Actual vs. Predicted Grades

A scatterplot compares actual and predicted final grades.

A diagonal reference line represents perfect prediction.

Points closer to the diagonal indicate smaller prediction errors.

### Residual Distribution

A histogram displays the difference between actual and predicted grades.

Residuals are calculated as:

`Residual = Actual Grade - Predicted Grade`

A vertical reference line at zero indicates no prediction error.

The distribution helps users examine the direction and spread of prediction errors.

Predictions are not clipped to the 0–20 grade range so that model errors remain visible.

## 12. Limitations

The project has several important limitations.

1. The dataset contains 395 student records from a specific educational setting, so findings may not generalize to other schools or populations.

2. Correlations and regression coefficients do not establish causal relationships.

3. Weekly study time is recorded in categories rather than exact hours.

4. Linear regression may not capture nonlinear relationships or interactions among predictors.

5. The models use selected numerical and ordinal variables rather than every available dataset feature.

6. Earlier grades (`G1` and `G2`) are not available at the beginning of the school year.

7. Recorded absences must be available at the intended prediction time to avoid using future information.

8. Cross-validation and held-out testing provide useful evaluation evidence but do not guarantee performance on new student populations.

9. Small groups created through interactive filtering may produce unstable descriptive statistics.

10. The application is intended for exploratory understanding and should not independently determine high-stakes academic decisions.

## 13. Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 14. Project Structure

```text
StudentPerformanceExplorer/
├── app.py
├── student-mat.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## 15. Running the Application Locally

Python 3.10 or newer is recommended.

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd StudentPerformanceExplorer
```

Replace `<repository-url>` with the actual GitHub repository URL.

### Step 2: Create a Virtual Environment

```bash
python3 -m venv .venv
```

### Step 3: Activate the Environment

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Run Streamlit

```bash
streamlit run app.py
```

The application will open in a browser.

Ensure that `student-mat.csv` is located in the same folder as `app.py`.

## 16. Deployment

The application can be deployed using Streamlit Community Cloud.

Deployment steps:

1. Upload the project files to a GitHub repository.
2. Open https://share.streamlit.io/.
3. Connect the GitHub repository.
4. Select the appropriate branch.
5. Set `app.py` as the main application file.
6. Deploy the application.
7. Verify that all pages, filters, visualizations, and regression models work.

**Live application:** Add the deployed Streamlit URL.

**GitHub repository:** Add the GitHub repository URL.

## 17. Reproducibility

The application uses a fixed random seed (`random_state=42`) for the train/test split and cross-validation.

Both regression models are evaluated using identical test observations.

The application includes the data source, required dependencies, modeling methods, and setup instructions.

This allows other users to reproduce the analysis using the provided dataset and application code.

## 18. AI Use Disclosure

I used ChatGPT (OpenAI, GPT-6) to help write and debug Streamlit code, improve the application interface, organize the model evaluation process, and refine project documentation.

I tested the application, reviewed the code, and checked the results to make sure everything worked correctly.

I made the final decisions about data preparation, visualizations, model selection, and interpretation of results.

All conclusions and limitations were reviewed and verified independently.

## 19. Citation

Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. Proceedings of FUture BUsiness TEChnology (FUBUTEC).

Dataset available from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/320/student+performance
