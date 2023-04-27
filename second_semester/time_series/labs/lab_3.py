import numpy as np
import pandas as pd
import seaborn as sns
from pprint import pprint
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from toolbox import backward_regression, auto_corr

np.random.seed(6313)

df = pd.read_csv("autos.clean.csv")
# print(df.head().to_string())
df = df[['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore', 'stroke',
         'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']]

y = df.price
x = df.drop(['price'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=6313)

# print(x_test.shape)
print(f"Shape of train set: {x_train.shape}, rows : {x_train.shape[0]}, columns: {x_train.shape[1]}")
print(f"Shape of test set: {x_test.shape}, rows : {x_test.shape[0]}, columns: {x_test.shape[1]}")
print("Shape of train set labels: ", y_train.shape)
print("Shape of test set labels: ", y_test.shape)

sns.heatmap(df.corr())
plt.tight_layout()
plt.show()

df_ = df.drop(
    ['city-mpg', 'highway-mpg', 'engine-size', 'horsepower', 'width', 'length', 'curb-weight', 'bore', 'wheel-base'],
    axis=1)
X = df.drop(['price'], axis=1)
H = np.dot(X.T, X)
s, d, v = np.linalg.svd(H, full_matrices=False)
print("Singular values: ", d)

condition_num = np.linalg.cond(X)
print("Condition number: ", condition_num)

scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
# x_test_s = scaler.transform(x_test)
# y_train_s = scaler.transform(y_train.values.reshape(-1, 1))
# y_test_s = scaler.transform(y_test.values.reshape(-1, 1))

# scaler = StandardScaler()
# # Fit and transform the data
# scaled_data = scaler.fit_transform(data)
# # Create a new dataframe using the scaled data and the original feature names
# df_scaled = pd.DataFrame(scaled_data, columns=data.columns)

x_one_tr = np.ones((x_train_s.shape[0], 1))  # , np.ones((x_test_s.shape[0], 1))
X_tr = np.hstack((x_one_tr, x_train_s))  # , np.hstack((x_one_ts, x_test_s))
# Y_tr = y_train_s, y_test_s

X, Y = X_tr, y_train
# b = (XTX)-1(XTY)
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
beta = [round(x, 2) for x in beta]
print("Unknown coefficients (beta): ", beta)

# print(x_train.columns)
model = sm.OLS(Y, X).fit()
print(round(model.params, 2))

# # fit the initial model
# model = sm.OLS(y, X).fit()
#
# # iterate until all features have a p-value less than 0.05
# while max(model.pvalues) > 0.05:
#     # remove the feature with the highest p-value
#     feature_to_remove = model.pvalues.idxmax()
#     X = X.drop(feature_to_remove, axis=1)
#
#     # fit the new model
#     model = sm.OLS(y, X).fit()
#
# print(model.summary())

feats = backward_regression(x_train_s, x_train, Y)
print("Selected features: ", feats.values)
print("Removed features: ", [x for x in x_train.columns.values if x not in feats.values])

new_X = x_train[feats]
new_X = scaler.fit_transform(new_X)
X = sm.add_constant(new_X)
final_model = sm.OLS(Y, X).fit()
print(final_model.summary())

x_test = x_test[feats]
x_test_s = scaler.transform(x_test)
X_test = sm.add_constant(x_test_s)
# print(X_test.shape)
y_pred = final_model.predict(X_test)

plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Price Forecasting')
plt.legend()
plt.show()

ryt = auto_corr(final_model.resid, lag=20, title="ACF", marker_thickness=4)

t_test = final_model.t_test(np.identity(len(final_model.params)))
print(t_test.summary())

f_test = final_model.f_pvalue
print(f_test)

# significant_level = 0.05
# non_sig_coeffs = [name for name, pval in final_model.t_test(X).pvalue_dict.items() if pval > significant_level]
# print("Non-significant coefficients:", non_sig_coeffs)
