import matplotlib.pyplot as plt
import numpy as np
np.random.seed(6202)
# -------------------------------------------------
p = np.array([[1, 1, 2, 2, -1, -2, -1, -2 ],
              [1, 2, -1, 0, 2, 1, -1, -2]])

t = np.array([[-1, -1, -1, -1, 1, 1, 1 , 1],
              [-1, -1, 1 ,1, -1, -1, 1, 1]])
# -------------------------------------------------

plt.scatter(p[0,:], p[1,:], label="input pattern")
plt.scatter(t[0,:], t[1,:], label="target")
plt.title("Patterns and Targets")
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()
# -------------------------------------------------
w = np.random.rand(2, 2)
b = np.random.rand(2,1)
alpha = 0.05
epochs = 1000
e = np.zeros((2, p.shape[1]))
sse = np.zeros((2, epochs))

for epoch in range(epochs):
    for i in range(p.shape[1]):
        # print(w)
        a = np.dot(w, p[:, i]).reshape(-1, 1) + b
        # print(a)
        # print(t[:, i])
        e[:, i] = np.array(t[:, i] - a.ravel())
        # print(e[:, i].reshape(-1, 1))
        w = w + 2 * alpha * np.dot(e[:, i].reshape(-1, 1), p[:, i].T.reshape(1, 2))
        b = b + 2 * alpha * np.array(e[:, i]).reshape(-1, 1)
    sse[:, epoch] = np.sum(e ** 2, axis=1)

plt.loglog(sse[0, :])
plt.loglog(sse[1, :])
plt.xlabel('Epochs')
plt.ylabel('SSE')
plt.title('Sum Squared Error vs Epochs log scaled')
plt.show()

# -------------------------------------------------

p1 = np.linspace(-2, 2, 100)
db1 = -w[0, 0] / w[0, 1] * p1 - b[0, 0] / w[0, 1]
db2 = -w[1, 0] / w[1, 1] * p1 - b[1, 0] / w[1, 1]

# Plotting decision boundaries
plt.scatter(p[0, :], p[1, :])
plt.plot(p1, db1)
plt.plot(p1, db2)
plt.ylim([-4, 4])
plt.xlabel('p1')
plt.ylabel('p2')
plt.title('Decision Boundaries')
plt.show()

# -------------------------------------------------

print("Perceptron produces straighter decision boundaries while Adaline's have slopes due to their learning algorithms. Adaline converges to a solution minimizing SSE but Perceptron may not converge or converge to an inferior solution in non-linearly separable data.")
