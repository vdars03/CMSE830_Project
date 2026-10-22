
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title='Student Performance Explorer', page_icon='🎓', layout='wide')
sns.set_theme(style='whitegrid')

@st.cache_data
def load_data():
    return pd.read_csv('student-mat.csv', sep=';')

@st.cache_data
def model_results(data):
    f1 = ['age', 'Medu', 'Fedu', 'studytime', 'failures', 'absences']
    f2 = f1 + ['G1', 'G2']
    tr, te = train_test_split(np.arange(len(data)), test_size=0.2, random_state=42)
    y = data['G3']
    out = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {'MAE': 'neg_mean_absolute_error', 'RMSE': 'neg_root_mean_squared_error', 'R2': 'r2'}
    for name, features in [('Without previous grades', f1), ('With previous grades', f2)]:
        x = data[features]
        model = LinearRegression().fit(x.iloc[tr], y.iloc[tr])
        pred = model.predict(x.iloc[te])
        scores = cross_validate(LinearRegression(), x.iloc[tr], y.iloc[tr], cv=cv, scoring=scoring)
        out[name] = {'features': features, 'pred': pred, 'actual': y.iloc[te].to_numpy(),
                     'MAE': mean_absolute_error(y.iloc[te], pred),
                     'RMSE': np.sqrt(mean_squared_error(y.iloc[te], pred)),
                     'R²': r2_score(y.iloc[te], pred), 'cv': scores}
    return out, len(tr), len(te)


def plot(data, x, kind='box', title=None, xlabel=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    if kind == 'bar':
        sns.barplot(data=data, x=x, y='G3', estimator='mean', errorbar=None, color='steelblue', ax=ax)
        ax.set_ylabel('Average final grade (0–20)')
    else:
        sns.boxplot(data=data, x=x, y='G3', color='lightblue', ax=ax)
        ax.set_ylabel('Final grade (0–20)')
    ax.set_xlabel(xlabel or x)
    ax.set_title(title or '')
    ax.set_ylim(0, 20)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def summary(data, group, label):
    t = data.groupby(group, observed=False)['G3'].agg(['count', 'mean', 'median', 'std']).reset_index()
    t.columns = [label, 'Students', 'Average grade', 'Median grade', 'Standard deviation']
    st.dataframe(t.round(2), use_container_width=True, hide_index=True)
    if (t['Students'] < 5).any():
        st.caption('Some groups have fewer than five students; interpret their averages cautiously.')


def correlation(data, col, label):
    r = data[col].corr(data['G3'], method='spearman') if data[col].nunique() > 1 and data['G3'].nunique() > 1 else np.nan
    st.metric(label, f'{r:.3f}' if pd.notna(r) else 'N/A')


def group_insight(data, group, label):
    t = data.groupby(group, observed=True)['G3'].agg(['count', 'mean']).dropna()
    t = t[t['count'] >= 5]
    if len(t) < 2:
        st.caption('Interpretation: Too few groups have at least five students for a reliable descriptive comparison.')
        return
    hi, lo = t['mean'].idxmax(), t['mean'].idxmin()
    diff = t.loc[hi, 'mean'] - t.loc[lo, 'mean']
    st.markdown(f"**What the chart shows:** Among groups with at least five students, {label} **{hi}** has the highest average final grade ({t.loc[hi, 'mean']:.2f}/20), compared with **{lo}** ({t.loc[lo, 'mean']:.2f}/20), a difference of {diff:.2f} points. These are descriptive differences, not causal effects.")


def correlation_insight(data, col):
    r = data[col].corr(data['G3'], method='spearman') if data[col].nunique() > 1 and data['G3'].nunique() > 1 else np.nan
    if pd.isna(r):
        st.caption('Interpretation: The current selection has too little variation to calculate a correlation.')
    else:
        direction = 'positive' if r > 0 else 'negative' if r < 0 else 'approximately zero'
        st.caption(f'Interpretation: Spearman ρ = {r:.3f} indicates a {direction} rank association in the selected students. Its magnitude describes association strength, not causation.')


df = load_data()
st.sidebar.title('🎓 Student Performance Explorer')
page = st.sidebar.radio('Navigation', ['Home', 'Explore Data', 'Analysis & Modeling', 'Data Preparation', 'Limitations & Documentation'])
st.sidebar.divider()
st.sidebar.caption('CMSE 830 | Student Performance Explorer')

if page == 'Home':
    st.title('🎓 Student Performance Explorer')
    st.write('Explore relationships between student characteristics and final mathematics grades, and compare two predictive models.')
    c1, c2, c3 = st.columns(3)
    c1.metric('Students', len(df))
    c2.metric('Variables', df.shape[1])
    c3.metric('Average final grade', f"{df['G3'].mean():.2f} / 20")
    st.subheader('Who is this for?')
    st.write('Students, educators, and academic advisors interested in patterns of academic performance.')
    st.subheader('How to use the dashboard')
    st.markdown('**Explore Data** contains interactive charts and group comparisons. **Analysis & Modeling** compares predictions with and without previous grades. **Data Preparation** documents quality checks and feature choices. **Limitations & Documentation** explains what the findings can and cannot show.')
    st.info('The data show associations, not proof that any student characteristic causes a grade change.')

elif page == 'Explore Data':
    st.title('Explore Data')
    st.write('Filter the students, then choose an analysis using the tabs below. All results on this page use the selected students.')
    c1, c2 = st.columns(2)
    with c1:
        a = st.slider('Maximum absences', 0, int(df['absences'].max()), int(df['absences'].max()))
    with c2:
        s = st.multiselect('Study-time categories', sorted(df['studytime'].unique().tolist()), default=sorted(df['studytime'].unique().tolist()))
    d = df.loc[(df['absences'] <= a) & df['studytime'].isin(s)].copy()
    c1, c2, c3 = st.columns(3)
    c1.metric('Students selected', len(d))
    c2.metric('Average final grade', f"{d['G3'].mean():.2f}" if not d.empty else 'N/A')
    c3.metric('Median final grade', f"{d['G3'].median():.2f}" if not d.empty else 'N/A')
    if d.empty:
        st.warning('No students match these filters. Select at least one study-time category or increase the absence limit.')
    else:
        d['absence_group'] = pd.cut(d['absences'], [-1, 0, 5, 10, np.inf], labels=['0', '1–5', '6–10', '11+'])
        tabs = st.tabs(['Overview', 'Study Time', 'Absences', 'Past Failures', 'Support', 'Correlations'])
        with tabs[0]:
            st.subheader('Final-grade distribution')
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(data=d, x='G3', bins=np.arange(-0.5, 21.5, 1), color='skyblue', ax=ax)
            ax.set_xlim(-0.5, 20.5)
            ax.set_xlabel('Final grade (0–20)')
            ax.set_ylabel('Students')
            st.pyplot(fig)
            plt.close(fig)
            st.caption('Why a histogram? It shows how frequently each grade occurs on the 0–20 scale, making clusters and unusually common scores easy to see.')
            st.markdown(f"**What the chart shows:** The most frequent final grade in this selection is {d['G3'].mode().iloc[0]:g}; the median is {d['G3'].median():.2f}/20. These values describe only the currently selected students.")
        with tabs[1]:
            st.subheader('Study time and final grades')
            summary(d, 'studytime', 'Study-time category')
            c1, c2 = st.columns(2)
            with c1: plot(d, 'studytime', 'bar', 'Average grade by study time', 'Study-time category')
            with c2: plot(d, 'studytime', 'box', 'Grade distribution by study time', 'Study-time category')
            correlation(d, 'studytime', 'Spearman correlation')
            group_insight(d, 'studytime', 'study-time category')
            correlation_insight(d, 'studytime')
            st.caption('Why these plots? Bars compare group averages; boxplots reveal spread, medians, and outliers that averages can hide.')
            st.caption('1 = <2 hours; 2 = 2–5 hours; 3 = 5–10 hours; 4 = >10 hours per week.')
        with tabs[2]:
            st.subheader('Attendance and final grades')
            summary(d, 'absence_group', 'Absence group')
            plot(d, 'absence_group', 'box', 'Grade distribution by absences', 'Absences')
            correlation(d, 'absences', 'Spearman correlation')
            group_insight(d, 'absence_group', 'absence group')
            correlation_insight(d, 'absences')
            st.caption('Why a boxplot? It compares grade distributions across absence ranges without implying absences caused the differences.')
        with tabs[3]:
            st.subheader('Previous class failures and final grades')
            summary(d, 'failures', 'Previous failures')
            c1, c2 = st.columns(2)
            with c1: plot(d, 'failures', 'bar', 'Average grade by past failures', 'Previous failures')
            with c2: plot(d, 'failures', 'box', 'Grade distribution by past failures', 'Previous failures')
            correlation(d, 'failures', 'Spearman correlation')
            group_insight(d, 'failures', 'previous-failure group')
            correlation_insight(d, 'failures')
            st.caption('Why both plots? Average-grade bars show group differences while boxplots show the variation within each group.')
        with tabs[4]:
            st.subheader('Family and school support')
            for col, label in [('famsup', 'Family educational support'), ('schoolsup', 'Extra school educational support')]:
                st.markdown(f'**{label}**')
                summary(d, col, 'Support (yes/no)')
                c1, c2 = st.columns(2)
                with c1: plot(d, col, 'bar', 'Average grade', label)
                with c2: plot(d, col, 'box', 'Grade distribution', label)
                group_insight(d, col, label)
            st.caption('Why these plots? Compare averages alongside within-group grade distributions. Support may be targeted toward students who are already struggling.')
            st.info('Students receiving support may already be struggling. Differences between groups cannot establish the effect of support.')
        with tabs[5]:
            st.subheader('Spearman correlation heatmap')
            cols = ['age', 'Medu', 'Fedu', 'studytime', 'failures', 'absences', 'G1', 'G2', 'G3']
            cr = d[cols].corr(method='spearman')
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.heatmap(cr, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1, ax=ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            gr = cr['G3'].drop('G3').sort_values(key=abs, ascending=False).round(3).reset_index()
            gr.columns = ['Variable', 'Spearman correlation with G3']
            st.dataframe(gr, use_container_width=True, hide_index=True)
            st.caption('Why a heatmap? A diverging color scale centered at zero makes positive and negative rank associations comparable. Each cell also displays its numerical value.')
            top = gr.dropna(subset=['Spearman correlation with G3']).head(1)
            if not top.empty:
                v = top.iloc[0]
                st.markdown(f"**What the chart shows:** **{v['Variable']}** has the strongest absolute Spearman association with final grade in this selection (ρ = {v['Spearman correlation with G3']:.3f}). This does not establish a causal relationship.")
            st.caption('G1 and G2 are previous grading-period results, not beginning-of-year information.')
        st.caption('All group differences and correlations are descriptive; they do not establish causation.')

elif page == 'Analysis & Modeling':
    st.title('Regression Modeling and Evaluation')
    st.write('Research question: How much do previous grades improve prediction of the final grade (G3)?')
    st.info('These models use the full dataset, not the Explore Data filters. Both use the same 80/20 train/test split (random_state=42).')
    results, ntr, nte = model_results(df)
    names = list(results)
    c1, c2 = st.columns(2)
    c1.metric('Training students', ntr)
    c2.metric('Test students', nte)
    st.subheader('Predictors')
    st.write('**Model 1 — Without previous grades:** ' + ', '.join(results[names[0]]['features']))
    st.write('**Model 2 — With previous grades:** ' + ', '.join(results[names[1]]['features']))
    st.caption('Absences must be known at prediction time. G1 and G2 are available only after their respective grading periods.')
    mt = pd.DataFrame([{'Model': name, 'MAE': r['MAE'], 'RMSE': r['RMSE'], 'R²': r['R²']} for name, r in results.items()])
    st.subheader('Held-out test results')
    st.dataframe(mt.round(3), use_container_width=True, hide_index=True)
    st.caption('MAE and RMSE: lower is better. R²: higher is better; negative values are possible.')
    a1, a2 = results[names[0]], results[names[1]]
    st.markdown(f"**Model comparison:** Adding previous grades changes test MAE from **{a1['MAE']:.2f}** to **{a2['MAE']:.2f}** grade points and test R² from **{a1['R²']:.3f}** to **{a2['R²']:.3f}**. These are results from the same held-out students, not proof that grades cause later performance.")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Model 1', 'Model 2'], mt['MAE'], color='steelblue')
        ax.set_ylabel('Test MAE')
        ax.set_title('Prediction error comparison')
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Model 1', 'Model 2'], mt['R²'], color='mediumseagreen')
        ax.set_ylabel('Test R²')
        ax.set_title('Explained variation comparison')
        st.pyplot(fig)
        plt.close(fig)
    st.caption('Why these bar charts? They directly compare the two models on the same test set; MAE expresses average prediction error in grade points, while R² summarizes explained variation.')
    st.subheader('Inspect one model')
    chosen = st.selectbox('Model', names)
    r = results[chosen]
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(r['actual'], r['pred'], alpha=0.65)
        ax.plot([0, 20], [0, 20], 'r--', label='Perfect prediction')
        ax.set_xlim(0, 20)
        ax.set_xlabel('Actual G3')
        ax.set_ylabel('Predicted G3')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.histplot(r['actual'] - r['pred'], bins=15, color='orange', ax=ax)
        ax.axvline(0, color='red', linestyle='--')
        ax.set_xlabel('Actual - predicted grade')
        ax.set_ylabel('Students')
        st.pyplot(fig)
        plt.close(fig)
    st.caption('Why these plots? Actual-versus-predicted points show closeness to the perfect-prediction diagonal; the residual histogram shows error direction and spread. Predictions are not clipped to 0–20, so model errors remain visible.')
    st.markdown(f"**Selected model:** {chosen} has test MAE **{r['MAE']:.2f}**, RMSE **{r['RMSE']:.2f}**, and R² **{r['R²']:.3f}**. Look for points far from the diagonal and residuals far from zero to identify prediction errors.")
    st.divider()
    st.subheader('Five-fold cross-validation on training data')
    ct = pd.DataFrame([{
        'Model': name,
        'Mean CV MAE': -r['cv']['test_MAE'].mean(),
        'SD CV MAE': (-r['cv']['test_MAE']).std(),
        'Mean CV RMSE': -r['cv']['test_RMSE'].mean(),
        'Mean CV R²': r['cv']['test_R2'].mean()
    } for name, r in results.items()])
    st.dataframe(ct.round(3), use_container_width=True, hide_index=True)
    st.caption('Cross-validation repeats evaluation across five training-data folds to show how model error varies with the validation sample; it does not use the held-out test set.')
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, r in results.items():
        ax.plot(range(1, 6), -r['cv']['test_MAE'], marker='o', label=name)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel('Fold')
    ax.set_ylabel('Validation MAE')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    if results[names[1]]['MAE'] < results[names[0]]['MAE']:
        st.success('Including previous grades lowers test MAE on this split.')
    else:
        st.info('Including previous grades does not lower test MAE on this split.')
    st.warning('Predictive relationships are not causal. This evaluation uses one held-out test set and cross-validation within the training set.')

elif page == 'Data Preparation':
    st.title('Data Preparation and Feature Engineering')
    c1, c2, c3 = st.columns(3)
    c1.metric('Original rows', len(df))
    c2.metric('Original columns', df.shape[1])
    c3.metric('Missing values', int(df.isna().sum().sum()))
    st.metric('Duplicate rows', int(df.duplicated().sum()))
    st.subheader('Missingness audit')
    ms = pd.DataFrame({'Variable': df.columns, 'Missing values': df.isna().sum().values, 'Missing %': (df.isna().mean().values * 100).round(2)})
    st.dataframe(ms, use_container_width=True, hide_index=True)
    st.subheader('Original data preview')
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader('Variable types and distinct values')
    vt = pd.DataFrame({'Variable': df.columns, 'Data type': df.dtypes.astype(str).values, 'Unique values': df.nunique().values})
    st.dataframe(vt, use_container_width=True, hide_index=True)
    st.subheader('Descriptive feature engineering')
    ed = df.copy()
    ed['absence_group'] = pd.cut(ed['absences'], [-1, 0, 5, 10, np.inf], labels=['0', '1–5', '6–10', '11+'])
    ed['studytime_group'] = ed['studytime'].map({1: '<2 hours', 2: '2–5 hours', 3: '5–10 hours', 4: '>10 hours'})
    ed['grade_group'] = pd.cut(ed['G3'], [-1, 9, 14, 20], labels=['Below 10', '10–14', '15–20'])
    st.dataframe(ed[['absences', 'absence_group', 'studytime', 'studytime_group', 'G3', 'grade_group']].head(10), use_container_width=True, hide_index=True)
    st.caption('Grade groups are descriptive, not prediction inputs or universal performance classifications.')
    st.dataframe(ed['grade_group'].value_counts(sort=False).rename_axis('Grade group').reset_index(name='Students'), use_container_width=True, hide_index=True)
    st.subheader('Model feature selection')
    features = pd.DataFrame({
        'Feature': ['age', 'Medu', 'Fedu', 'studytime', 'failures', 'absences', 'G1', 'G2'],
        'Description': ['Student age', "Mother's education", "Father's education", 'Weekly study-time category', 'Previous class failures', 'Recorded absences', 'First-period grade', 'Second-period grade'],
        'Model 1': ['Yes'] * 6 + ['No', 'No'],
        'Model 2': ['Yes'] * 8
    })
    st.dataframe(features, use_container_width=True, hide_index=True)
    st.subheader('Preprocessing decisions')
    st.markdown('**Missing values:** No imputation is needed for this dataset.  \n**Duplicates:** Checked and reported; records are not silently dropped.  \n**Predictors:** Selected numeric/ordinal variables; categorical text columns are not used in these models.  \n**Scaling:** Ordinary least-squares predictions do not require feature scaling.  \n**Target:** G3 is never a predictor. Engineered grade categories are descriptive only.  \n**Evaluation:** Identical 80/20 split for both models; five-fold cross-validation on training data only.')

elif page == 'Limitations & Documentation':
    st.title('Limitations & Documentation')
    st.subheader('Important limitations')
    st.markdown('- Small sample from a specific educational context; results may not generalize to other schools.\n- Correlation and model coefficients do not establish causation.\n- Study time is categorical, not measured in exact hours.\n- Linear regression is limited to additive, linear predictor effects.\n- Only selected numerical and ordinal predictors are used.\n- G1 and G2 are unavailable for beginning-of-year prediction.\n- Absence totals must be known at prediction time to avoid using future information.\n- Five-fold cross-validation does not guarantee performance in other student populations.')
    st.subheader('Data source')
    st.markdown('[UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance) — mathematics course (`student-mat.csv`).')
    st.subheader('Methodology')
    st.write('The app presents filtered exploratory analysis, descriptive engineered categories, two linear regression models, held-out evaluation, and training-only five-fold cross-validation.')
    st.subheader('Project documentation')
    st.write('For setup, dependencies, reproducibility steps, and the AI-use disclosure, see the project README in the GitHub repository.')
