import numpy as np
import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_openml
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve

# Load the MNIST dataset from Keras
# mnist = fetch_openml('mnist_784')
# X, y = mnist['data'], mnist['target']
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Plot a few images
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(8, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28), cmap='binary')
    ax.set_title(f"Label: {y[i]}")
plt.tight_layout()
plt.show()

# Shuffle the data and split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
np.random.seed(42)
shuffle_index = np.random.permutation(len(X_train))
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Train an MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, solver='sgd', learning_rate_init=0.1, random_state=42)
train_sizes, train_scores, valid_scores = learning_curve(mlp, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=3)

# Plot the learning curve
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color="g")
plt.legend(loc="best")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.show()

# Fit the classifier to the training data
mlp.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp.predict(X_test)

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.matshow(cm, cmap=plt.cm.gray)
plt.show()

# Show the classification report
cr = classification_report(y_test, y_pred)
print(cr)
