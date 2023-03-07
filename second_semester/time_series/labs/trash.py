# import matplotlib.pyplot as plt
# import numpy as np
#
# # Define the data for the subplots
# x = np.linspace(0, 2 * np.pi, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)
# y3 = np.tan(x)
# data = [y1, y2, y3]
#
# # Create the figure and subplots
# fig, axes = plt.subplots(nrows=3, ncols=2)
#
# # Loop over the subplots and fill each one with data
# for i, ax in enumerate(axes.flat):
#     if i % 2 == 0:
#         ax.plot(x, data[i//2])
#         ax.set_title(f"Plot {i//2+1}")
#     else:
#         ax.hist(data[i//2], bins=10)
#         ax.set_title(f"Histogram {i//2+1}")
#
# # Add a title to the figure
# fig.suptitle("Subplots Example")
#
# # Display the figure
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

def lms_algorithm(inputs, targets, eta, n_iterations):

    # Initialize weights with zeros
    weights = np.zeros((inputs.shape[1], 1))
    # Perform iterations
    for i in range(n_iterations):
        # Calculate prediction and error
        prediction = np.dot(inputs, weights)
        error = targets - prediction
        # Update weights
        weights += eta * np.dot(inputs.T, error)
    return weights

# Input patterns and targets
inputs = np.array([[1, 1], [-1, -1]])#, [-1, 1], [1, -1]])
targets = np.array([[1], [-1]])#, [-1], [-1]])

# Apply LMS algorithm
eta = 0.05
n_iterations = 40
weights = lms_algorithm(inputs, targets, eta, n_iterations)

# Plot decision boundary
x = np.linspace(-2, 2, 100)
y = -(weights[0] * x) / weights[1]
plt.plot(x, y, label='Decision boundary')

# Plot input patterns
for i in range(inputs.shape[0]):
    if targets[i] == 1:
        plt.scatter(inputs[i, 0], inputs[i, 1], marker='o', color='r')
    else:
        plt.scatter(inputs[i, 0], inputs[i, 1], marker='x', color='b')

plt.legend()
plt.show()
