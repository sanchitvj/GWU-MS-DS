import matplotlib.pyplot as plt
import numpy as np
np.random.seed(6201)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def delta_sigmoid(x):
    return - np.exp(-x) / ((1 + np.exp(-x)) ** 2)


s1, r = int(input("Enter #neurons s1: ")), 1
epochs = int(input("Enter #epochs: "))
alpha = float(input("Enter alpha: "))
w1 = np.random.uniform(-0.5, 0.5, size=(s1, r))
b1 = np.random.uniform(-0.5, 0.5, size=(s1, 1))
w2 = np.random.uniform(-0.5, 0.5, size=(s1, 1))
b2 = np.random.uniform(-0.5, 0.5, size=(1, 1))

input_p = np.linspace(-2, 2, num=100)
# p = np.random.choice(input_p, size=(r, 1))
# target = np.exp(np.abs(p)) * np.sin(np.pi * p)


def g_of_p(p):
    return np.exp(np.abs(p)) * np.sin(np.pi * p)


def forward(w1, b1, w2, b2, p):
    n1 = w1.dot(p) + b1
    a1 = sigmoid(n1)
    # a1 = sigmoid(np.dot(p.reshape(-1, 1), w1.T) + b1)
    n2 = w2.T.dot(a1) + b2
    a2 = n2

    return n1, a1, n2, a2


def backward(n1, a1, n2, a2, w1, b1, w2, b2, p, error, alpha):

    sM = -2 * error# * delta_sigmoid(n2)
    sm = w2.dot(sM) * delta_sigmoid(n1)

    w2 -= alpha * sM.dot(a1.T).T
    b2 -= alpha * sM
    w1 -= alpha * sm.dot(p.T)
    b1 -= alpha * sm

    return w1, b1, w2, b2, error


def train_nn(w1, b1, w2, b2, input_p, alpha):
    sse = []
    y_pred = []
    for i in range(epochs):
        errors, y = [], []
        for p in input_p:
            target = g_of_p(p)
            n1, a1, n2, a2 = forward(w1, b1, w2, b2, p)
            error = target - a2
            if i == epochs - 1:
                y_pred.append(a2.squeeze())
            w1, b1, w2, b2, error = backward(n1, a1, n2, a2, w1, b1, w2, b2, p, error, alpha)
            errors.append(error**2)

        sse.append(np.sum(errors))

    return sse, y_pred


sse, y_pred = train_nn(w1, b1, w2, b2, input_p, alpha)

# y_pred = []
# for i in input_p:
#     # w1 = np.random.uniform(-0.5, 0.5, size=(s1, r))
#     # b1 = np.random.uniform(-0.5, 0.5, size=(s1, 1))
#     # w2 = np.random.uniform(-0.5, 0.5, size=(s1, 1))
#     # b2 = np.random.uniform(-0.5, 0.5, size=(1, 1))
#     n1, a1, n2, a2 = forward(w1, b1, w2, b2, i)
#     y_pred.append(a2.squeeze())
# print(y_pred)
# y = np.exp(np.abs(input_p)) * np.sin(np.pi * input_p)

plt.plot(input_p, g_of_p(input_p), 'r-', label='True Function')
plt.plot(input_p, y_pred, 'b-', label='Predicted Function')
plt.xlabel('p')
plt.ylabel('g(p)')
plt.grid()
plt.title('Comparison of True and Predicted Functions')
plt.legend()
plt.show()

plt.plot([int(i) for i in range(epochs)], sse)
plt.xlabel("Epochs")
plt.ylabel("SSE")
plt.title('Sum Squared Errors')
plt.show()