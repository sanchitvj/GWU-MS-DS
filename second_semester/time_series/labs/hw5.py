import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toolbox import ar_ma_dlsim

np.random.seed(6313)


# ARMA (2,1):  y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t) - 0.5e(t-1)
N = int(input("Enter number of samples (N): "))
na = int(input("Enter AR order: "))
nb = int(input("Enter MA order: "))

den, num = [], []
print("Enter the coefficients of AR (separated by space):")
den = [float(x) for x in input().split()]
print("Enter the coefficients of MA (separated by space):")
num = [float(x) for x in input().split()]

# N = 10000
# num, den = [1, 0.5, -0.4], [1, 0.5, 0.2]
# na, nb = 2, 2
system = (num, den, 1)
e = np.random.normal(0, 1, N)
y = ar_ma_dlsim(e, num, den, "ar")

true_theta = [i for i in den[1:na+1]] + [i for i in num[1:nb+1]]
# true_theta = np.array([den[1], den[2], num[1]])
print("True coefficients: ", true_theta)
true_theta = np.array(true_theta)


def arma_model(theta, y, e, na, nb):
    # np.random.seed(6313)
    T = len(y)
    n = max(na, nb)
    y_pred = np.zeros(T)
    e_hat = np.zeros(T)
    num, den = [1] + theta[n:].tolist(), [1] + theta[:n].tolist()
    num = num if len(num) == n+1 else num + [0] * (n+1-len(num))  # [i for i in np.zeros(n+1-len(num))]
    den = den if len(den) == n+1 else den + [0] * (n+1-len(den))  # [i for i in np.zeros(n+1-len(den))]

    assert len(num) == len(den), f"Length of num {num} must be equal to length of den {den}."
    # y_pred = ar_ma_dlsim(y, den, num, "ar")
    # e_hat = y_pred
    y_pred = ar_ma_dlsim(e, num, den, "ar")
    for t in range(0, T):
        # y_pred[t] = -theta[0] * y[t - 1] - theta[1] * y[t - 2] + theta[2] * e[t - 1] + e[t]
        e_hat[t] = y[t] - y_pred[t]

    return y_pred, e_hat


# Define the Levenberg-Marquardt algorithm
def levenberg_marquardt(y, theta0, e, mu0=0.01, mu_max=1e10, epsilon=1e-3, h=1e-6, max_iter=100):
    # np.random.seed(6313)
    N = len(y)
    n = len(theta0)
    theta = theta0
    mu = mu0
    iterations = 0
    sse_new, iters = [], []
    while iterations < max_iter:
        iterations += 1
        y_pred, e_hat = arma_model(theta, y, e, na, nb)
        J = np.zeros((N, n))
        for i in range(n):
            params_perturb = theta  # np.array(params)
            params_perturb[i] += h
            _, residuals_perturb = arma_model(params_perturb, y, e, na, nb)  # model_func(params_perturb, data)
            J[:, i] = (e_hat.reshape(-1) - residuals_perturb.reshape(-1)) / h

        A = J.T.dot(J)
        A += mu * np.identity(n)
        delta_theta = np.linalg.inv(A).dot(J.T.dot(e_hat))  # np.linalg.solve(A, J.T @ e_hat)
        theta_new = theta + delta_theta
        y_pred_new, e_hat_new = arma_model(theta_new, y, e, na, nb)
        SSE_new = np.sum(e_hat_new ** 2)
        sse_new.append(SSE_new)
        SSE = np.sum(e_hat ** 2)
        if SSE_new < SSE:
            if np.linalg.norm(delta_theta) < epsilon * np.linalg.norm(theta):
                sigma_e_squared_hat = SSE_new / (N - n)
                cov_theta_hat = sigma_e_squared_hat * np.linalg.inv(A)
                print(f"Converged in {iterations} iterations.")
                iters.append(iterations)
                return theta_new, sigma_e_squared_hat, cov_theta_hat, sse_new, iters
            else:
                theta = theta_new
                mu /= 10
        else:
            mu *= 10
            if mu > mu_max:
                print("Failed to converge. Maximum mu reached.")
                return theta, None, None

    print("Failed to converge. Maximum iterations reached.")
    return theta, None, None


theta0 = np.zeros(len(true_theta), dtype=np.float64)
theta_hat, sigma_e_squared_hat, cov_theta_hat, sse_new, iters = levenberg_marquardt(y, theta0, e)

plt.plot(sse_new)
plt.title("SSE vs. Iterations")
plt.xlabel("Iterations")
plt.ylabel("SSE")
plt.show()

zeros = np.roots(theta_hat[:na])
poles = np.roots(theta_hat[na:])
print("Zeros: ", zeros)
print("Poles: ", poles)

if sigma_e_squared_hat is not None:
    for i in range(na+nb):
        process = "AR"
        if i >= na:
            process = "MA"
        print(f"Estimated {process} coeff{i}: {(theta_hat[i]):.3f}")
        print(f"Confidence interval {process} coeff{i}: ",
              [true_theta[i] - 2 * np.sqrt(cov_theta_hat[i, i]), true_theta[i] + 2 * np.sqrt(cov_theta_hat[i, i])])

    print("Covariance matrix of estimated coefficients:\n", cov_theta_hat)
    print(f"Estimated error variance: {sigma_e_squared_hat:.4f}")
