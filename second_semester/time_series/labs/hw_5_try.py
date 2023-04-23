import numpy as np
from scipy.optimize import least_squares


def arma22_residuals(params, data):
    a1, a2, b1, b2 = params
    residuals = np.zeros_like(data)

    for t in range(4, len(data)):
        residuals[t] = (data[t] - a1 * data[t - 1] - a2 * data[t - 2] -
                        b1 * residuals[t - 1] - b2 * residuals[t - 2])

    return residuals


def arma22_sse(params, data):
    residuals = arma22_residuals(params, data)
    sse = np.sum(residuals ** 2)
    return sse


def estimate_arma22(data, max_iter=100, epsilon=1e-3, mu_max=1e10):
    theta = np.random.normal(size=4)  # Initialize with random values
    mu = 0.01
    num_iterations = 0

    while num_iterations < max_iter:
        num_iterations += 1
        jacobian = approx_jacobian(theta, data, arma22_residuals)
        A = jacobian.T @ jacobian
        g = jacobian.T @ arma22_residuals(theta, data)
        theta_new = least_squares(lambda d: A @ d - g, np.zeros_like(theta)).x

        sse_theta = arma22_sse(theta, data)
        sse_theta_new = arma22_sse(theta_new, data)

        if sse_theta_new < sse_theta:
            if np.linalg.norm(theta_new - theta) < epsilon:
                theta = theta_new
                sse = sse_theta_new
                N = len(data)
                n = len(theta)
                sigma_e2 = sse / (N - n)
                cov_theta = sigma_e2 * np.linalg.inv(A)
                return theta, cov_theta
            else:
                theta = theta_new
                mu /= 10
        else:
            while sse_theta_new >= sse_theta:
                mu *= 10
                if mu > mu_max:
                    print("Error: mu exceeded mu_max")
                    return theta, None

                A += mu * np.eye(A.shape[0])
                theta_new = least_squares(lambda d: A @ d - g, np.zeros_like(theta)).x
                sse_theta_new = arma22_sse(theta_new, data)


def approx_jacobian(theta, data, func, eps=1e-6):
    """Approximate the Jacobian matrix of func with respect to theta."""
    jacobian = np.zeros((len(data), len(theta)))
    for i, t in enumerate(theta):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        theta_minus = theta.copy()
        theta_minus[i] -= eps

        jacobian[:, i] = (func(theta_plus, data) - func(theta_minus, data)) / (2 * eps)

    return jacobian


# Generate synthetic ARMA(2,2) data
np.random.seed(42)
n_samples = 1000
a1, a2, b1, b2 = 0.6, -0.4, 0.5, -0.3
noise = np.random.normal(0, 1, size=n_samples)
data = np.zeros_like(noise)

for t in range(4, n_samples):
        data[t] = (a1 * data[t-1] + a2 * data[t-2] +
                   b1 * noise[t-1] + b2 * noise[t-2] + noise[t])

# Call estimate_arma22 to estimate ARMA(2,2) parameters
estimated_params, estimated_cov = estimate_arma22(data)

# Print the estimated parameters and covariance
print(f"Estimated parameters: {estimated_params}")
print(f"Estimated covariance: \n{estimated_cov}")