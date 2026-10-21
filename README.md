# Student Performance Explorer

An interactive Streamlit dashboard examining associations between student characteristics, study habits, academic history, and final mathematics grades. Built as a CMSE 830 data science project.

## Audience and questions

For students, educators, and academic advisors interested in exploratory patterns, not individual student decisions. Questions include how study time, absences, prior failures, and educational support relate to final grades, and how much earlier grades improve predictive accuracy.

## Dataset

- Source: [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez and Silva, 2008).
- File: `student-mat.csv`, semicolon-delimited; 395 records, 33 original variables.
- Target: `G3`, final mathematics grade on a 0–20 scale.
- `G1` and `G2` are first- and second-period grades.

Include the original dataset file in the repository and retain its attribution. The dashboard requires the CSV in the same folder as `app.py`.

## Features

- Interactive filtering by maximum absences and study-time category.
- Grade distributions and group comparisons for study time, absences, prior failures, family support, and school support.
- Spearman correlation analysis and heatmap.
- Two linear regression models: one without `G1`/`G2`, and one with them.
- Fixed 80/20 train/test split (`random_state=42`), test MAE, RMSE, and R², and five-fold cross-validation on the training portion.
- Data-quality checks, descriptive feature engineering, and limitations documentation.

## Run locally

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

## Deployment

Push `app.py`, `student-mat.csv`, `requirements.txt`, and this README to a GitHub repository. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from the repository, choose the branch containing `app.py`, and set the main file path to `app.py`. Verify that every page loads and the models run on the deployed site.

**Live app:** Add deployment URL after publishing.

## Data preparation and methods

The app checks missing values and duplicates, reports column types, and creates descriptive categories for absences, study time, and final grades. These engineered categories are for exploration, not regression inputs. Regression predictors are `age`, `Medu`, `Fedu`, `studytime`, `failures`, `absences`, with `G1` and `G2` added in the second model. No imputation is used when the dataset has no missing values. Both models are evaluated on the same held-out test observations; five-fold cross-validation uses only the training subset.

## Limitations and responsible interpretation

- The data represent a small, specific educational setting; findings may not generalize.
- Correlations and regression coefficients do not establish causation.
- Study time is categorical, not an exact number of hours.
- `G1` and `G2` are unavailable at the start of the year. Absences must also be measured before any claimed prediction date.
- A linear model may miss nonlinear relationships and interactions.
- Predictions should not be used alone to make high-stakes decisions about students.

## AI use disclosure

Generative AI (ChatGPT) assisted with drafting and refining Streamlit code, model-evaluation workflow, interface text, and project documentation. The student ran the application, checked the outputs and behavior, and remains responsible for verifying the analysis, interpreting the results, and reporting limitations. Edit this disclosure to accurately reflect your own use and verification.

## Citation

Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. Proceedings of FUture BUsiness TEChnology (FUBUTEC).
