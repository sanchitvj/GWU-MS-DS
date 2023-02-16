import matplotlib.pyplot as plt
import numpy as np

# Define the data for the subplots
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
data = [y1, y2, y3]

# Create the figure and subplots
fig, axes = plt.subplots(nrows=3, ncols=2)

# Loop over the subplots and fill each one with data
for i, ax in enumerate(axes.flat):
    if i % 2 == 0:
        ax.plot(x, data[i//2])
        ax.set_title(f"Plot {i//2+1}")
    else:
        ax.hist(data[i//2], bins=10)
        ax.set_title(f"Histogram {i//2+1}")

# Add a title to the figure
fig.suptitle("Subplots Example")

# Display the figure
plt.show()
