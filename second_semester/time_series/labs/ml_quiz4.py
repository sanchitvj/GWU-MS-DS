# Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# Defining input and target patterns
p = np.array([[1, 1, 2, 2, -1, -2, -1, -2],
              [1, 2, -1, 0, 2, 1, -1, -2]])
t = np.array([[-1, -1, -1, -1, 1, 1, 1, 1],
              [-1, -1, 1, 1, -1, -1, 1, 1]])

# Plotting input and target patterns
plt.scatter(p[0, :], p[1, :], c='b')
plt.scatter(t[0, :], t[1, :], c='y')
plt.legend(['Input', 'Target'])
plt.grid()
plt.show()

# Initializing weights, biases, learning rate, and epochs
W = np.random.rand(2, 2)
b = np.random.rand(2, 1)
alpha = 0.04
epochs = 1000

# Initializing error and SSE arrays
e = np.zeros((2, p.shape[1]))
SSE = np.zeros((2, epochs))

# Training the network
for j in range(epochs):
    for i in range(p.shape[1]):
        a = np.dot(W, p[:, i]).reshape(-1, 1) + b
        e[:, i] = np.array(t[:, i] - a.ravel())
        W = W + 2 * alpha * np.dot(e[:, i].reshape(-1, 1), p[:, i].T.reshape(1, 2))
        b = b + 2 * alpha * np.array(e[:, i]).reshape(-1, 1)
    SSE[:, j] = np.sum(e ** 2, axis=1)

# Plotting SSE vs epochs
plt.figure()
plt.loglog(SSE[0, :])
plt.loglog(SSE[1, :])
plt.grid()
plt.xlabel('Epochs (log scale)')
plt.ylabel('SSE (log scale)')
plt.title('Sum Squared Error vs Epochs')
plt.tight_layout()
plt.show()

# Calculating decision boundaries
p1 = np.linspace(-2, 2, 100)
DB1 = -W[0, 0] / W[0, 1] * p1 - b[0, 0] / W[0, 1]
DB2 = -W[1, 0] / W[1, 1] * p1 - b[1, 0] / W[1, 1]

# Plotting decision boundaries
plt.figure()
plt.scatter(p[0, :], p[1, :])
plt.plot(p1, DB1)
plt.plot(p1, DB2)
plt.xlim([-2.5, 2.5])
plt.ylim([-2.5, 2.5])
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Decision Boundaries')
plt.grid()
plt.tight_layout()
plt.show()