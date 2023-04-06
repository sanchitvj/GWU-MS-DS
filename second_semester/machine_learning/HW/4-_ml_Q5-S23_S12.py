import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report, confusion_matrix
import warnings, gc
warnings.filterwarnings('ignore')
np.random.seed(6202)
# ---------------------Load Dataset-------------------------------------------------------------------------------------

df = fetch_openml("mnist_784")
print("Dataset loaded")
X, y = df.data, df.target

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# print(x_train[0])
# ---------------------Plot a sample Data-------------------------------------------------------------------------------

# fig, axes = plt.subplots(nrows=1, ncols=2)
# for i in range(2):
#     axes[i].imshow(x_train[i].reshape(28, 28), cmap=plt.cm.gray)
#     axes[i].set_title("Label: {}".format(y_train[i]))
# plt.show()
# plt.plot(x[1].reshape(28, 28))
# plt.show()

indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
mlp.fit(X_train, y_train)

# Show the confusion matrix and classification metric
y_pred = mlp.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Confusion matrix:\n", conf_mat)
print("\nClassification report:\n", class_report)


# ---------------------Train-------------------------------------------------------------------------------
# shuffle_index = np.random.permutation(len(x_train))
# x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
#
# # Train a MLP classifier
# clf = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=20, solver='sgd', verbose=10, random_state=42)
# clf.fit(X_train, y_train)

# ---------------------Learning Curve-------------------------------------------------------------------------------
y_pred = clf.predict(X_test)
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

train_sizes, train_scores, test_scores = learning_curve(
    mlp, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy')
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 5))
plt.title("Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("Accuracy Score")
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")
plt.legend(loc="best")
plt.show()

gc.collect()