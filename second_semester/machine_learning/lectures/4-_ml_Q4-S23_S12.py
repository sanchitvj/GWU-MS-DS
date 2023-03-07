import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# -------------------------------------------------
p = np.array([[1, 1, 2, 2, -1, -2, -1, -2 ],
              [1, 2, -1, 0, 2, 1, -1, -2]])

t = np.array([[-1, -1, -1, -1, 1, 1, 1 , 1],
              [-1, -1, 1 ,1, -1, -1, 1, 1]])
# -------------------------------------------------
# sns.scatterplot(p, t)
# plt.scatter(p, t)
plt.scatter(p[0], p[1], c=t[1], cmap='bwr')
plt.title("Patterns and Targets")
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()


# -------------------------------------------------
# w = np.zeros((2, 1))
# alpha = 0.05
# mse = 0
# mse_ls = []
# epochs = 2
#
# for i in range(3):
#     for j in range(p.shape[1]):
#         y = np.dot(w.T, p[:, j])
#         e = t[:, j] - y
#         w = w + alpha * np.outer(e, p[:, j])
#         mse += (y - e) ** 2
#     mse_ls.append(mse)
#
# # Print final weights
# print("Final weights:")
# print(w)
# print(mse_ls)
class Adaline:
    def __init__(self, input_size):
        self.w = np.zeros(input_size)
        self.b = 0

    def forward(self, x):
        return np.dot(self.w, x) + self.b

    def update(self, x, y, eta):
        e = y - self.forward(x)
        self.w += eta * e * x
        self.b += eta * e

        return  self.w, self.b

# Train the ADALINE network
net = Adaline(input_size=2)
eta = 0.01
epochs = 4
mse_list = []

for epoch in range(epochs):
    mse = 0
    for i in range(p.shape[1]):
        x = p[:, i]
        y = t[0, i]
        w, b = net.update(x, y, eta)
        mse += (y - net.forward(x))**2
    mse_list.append(mse)

print(mse_list)
plt.plot(mse_list)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()


# -------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(p[0, t[0, :] == 1], p[1, t[0, :] == 1], c='b', label='Class 1')
ax.scatter(p[0, t[0, :] == -1], p[1, t[0, :] == -1], c='r', label='Class -1')
ax.set_xlabel('p1')
ax.set_ylabel('p2')
ax.set_title('Decision Boundary')
ax.legend()

# Decision boundary equation
slope = -w[0] / w[1]
intercept = -b / w[1]
x = np.array([-3, 3])
y = slope * x + intercept
print(y)
ax.plot(x, y, c='k', label='Decision boundary')

plt.show()


# -------------------------------------------------




