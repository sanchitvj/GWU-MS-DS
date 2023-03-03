import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
np.random.seed(6313)

df = pd.read_csv("autos.clean.csv")
# print(df.head().to_string())
df = df[['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']]

y = df.price
x = df.drop(['price'], axis=1)

x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

print(x_test)
print("Shape of train set: ", x_train.shape)
print("Shape of test set: ", y_train.shape)
print("Shape of train set labels: ", x_test.values.reshape(-1, 1).shape)
print("Shape of test set labels: ", y_test.shape)

sns.heatmap(df.corr())
plt.tight_layout()

df_ = df.drop(['city-mpg', 'highway-mpg', 'engine-size', 'horsepower', 'width', 'length', 'curb-weight', 'bore', 'wheel-base'], axis=1)
s, d, v = np.linalg.svd(df_, full_matrices=False)
print("Singular values: ", d)

condition_num = np.max(d) / np.min(d)
print("Condition number: ", condition_num)

scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
y_train_s = scaler.fit_transform(y_train)
x_test_s = scaler.transform(x_test.values.reshape(-1, 1))
y_test_s = scaler.transform(y_test.values.reshape(-1, 1))

x_one_tr, x_one_ts = np.ones((x_train_s.shape[0], 1)), np.ones((x_test_s.shape[0], 1))
X_tr, X_ts = np.hstack((x_one_tr, x_train_s)), np.hstack((x_one_ts, x_test_s))
Y_tr, Y_ts = y_train_s, y_test_s

# b = (XTX)-1(XTY)
# beta = np.linalg.inv(X.T.dot(X))

plt.show()