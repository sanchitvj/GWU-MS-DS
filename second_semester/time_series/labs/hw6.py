import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import normaltest
from lifelines import KaplanMeierFitter
from sklearn.impute import SimpleImputer

np.random.seed(6301)
df = pd.read_csv("https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head().to_string())
print(df.describe())
print("#observations: ", df.shape[0])
print("#features: ", df.shape[1])

cat_features = df.select_dtypes(include='object')
print("Categorical Features:")
print(cat_features.columns.values)

num_features = df.select_dtypes(include=['int64', 'float64'])
print("\nNumerical Features:")
print(num_features.columns.values)

df['TotalCharges'] = df['TotalCharges'].str.replace(' ', '')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='raise')
print(df.TotalCharges.head())

df["Churn"] = df["Churn"].astype('category')
df["Churn"] = df.Churn.cat.codes
print(df.Churn.head())

print(df.isna().sum())

imputer = SimpleImputer(strategy='median')
df['TotalCharges'] = imputer.fit_transform(df['TotalCharges'].values.reshape(-1, 1))

durations = df['tenure']
event_observed = df['Churn']

kmf = KaplanMeierFitter()
kmf.fit(durations, event_observed, label="estimated survival curve")
kmf.plot_survival_function()
plt.ylabel("Probability")
plt.show()

cohorts = ["Month-to-month", "Two year", "One year"]
for cohort in cohorts:
    cohort_data = df[df['Contract'] == cohort]
    kmf.fit(cohort_data['tenure'], event_observed=cohort_data['Churn'], label=cohort)
    # kmf.plot(ci_show=False)
    kmf.plot_survival_function()

plt.title('Kaplan-Meier Curves by Contract')
plt.legend(title='Contract')
plt.ylabel("Probability")
plt.show()

cohorts = ["Yes", "No"]
for cohort in cohorts:
    cohort_data = df[df['StreamingTV'] == cohort]
    kmf.fit(cohort_data['tenure'], event_observed=cohort_data['Churn'], label=cohort)
    # kmf.plot(ci_show=False)
    kmf.plot_survival_function()

plt.title('Kaplan-Meier Curves by Streaming TV or not')
plt.legend(title='StreamingTV')
plt.ylabel("Probability")
plt.show()


df2 = pd.read_csv("https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/dd.csv")

print(df2.head().to_string())

cohorts = ["Asia", "Europe", "Africa", "Americas", "Oceania"]
for cohort in cohorts:
    cohort_data = df2[df2['un_continent_name'] == cohort]
    kmf.fit(cohort_data['duration'], event_observed=cohort_data['observed'], label=cohort)
    kmf.plot_survival_function()

plt.title('Survival for the political regimes based on different regions')
plt.legend(title='Regions')
plt.ylabel("Probability")
plt.show()

# Interpretation: The Kaplan-Meier curves show the estimated survival probabilities
# over time for the three cohorts ("Month-to-month", "Two year", and "One year").
# The higher the survival probability, the better the retention. From the plot, we
# can observe that the "Two year" cohort has the highest survival probability,
# indicating better retention compared to the other cohorts. The "One year" cohort
# has intermediate retention, and the "Month-to-month" cohort has the lowest
# retention as evidenced by the steepest decline in survival probability over time.
# This suggests that the length of the contract (i.e., the cohort) may affect
# customer retention, with longer contract lengths generally associated with
# better retention.




