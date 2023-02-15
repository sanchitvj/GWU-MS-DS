import matplotlib.pyplot as plt
import numpy as np

# Create sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, -3, 4, -5, 6])

# Plot the stem plot with both positive and negative values
plt.stem(x, y, bottom=-6)

# Add labels and title
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Stem Plot with Both Positive and Negative Values')

plt.show()
