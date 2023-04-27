import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
np.random.seed(6301)

df = pd.read_csv("question1.csv")
print(df.shape)
print(df.head().to_string())
df = df.drop(["Unnamed: 0"], axis=1)
print(df.isna().sum())

kmf = KaplanMeierFitter()
kmf.fit(df['time'], df["delta"], label="estimated survival curve")
kmf.plot_survival_function()
plt.ylabel("Probability")
plt.title("KM curve")
plt.show()

tumors = ["Aneuploid", "Diploid"]
tumor_type = [1, 2]
for tumor, value in zip(tumors, tumor_type):
    cohort_data = df[df['type'] == value]
    kmf.fit(cohort_data['time'], event_observed=cohort_data['delta'], label=tumor)
    # kmf.plot(ci_show=False)
    kmf.plot_survival_function()

plt.title('Kaplan-Meier Curves by tumor type')
plt.legend(title='Tumor type')
plt.ylabel("Probability")
plt.show()

df_ = df.copy()
df_ = df_[df_["time"] < 51]
kmf = KaplanMeierFitter()
kmf.fit(df_['time'], df_["delta"], label="estimated survival curve")
kmf.plot_survival_function()
# kmf.probability()
plt.ylabel("Probability")
plt.show()