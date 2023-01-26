import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

x = np.random.normal(0, 1, 1000)

date = pd.date_range(start='1/1/2000',
                     end='12/31/2000',
                     periods=len(x))

t = np.linspace(-np.pi, np.pi, len(x))
y = 5 * np.sin(t) + x
df = pd.DataFrame(y, columns=['temp'])
df.index = date

df.plot()
plt.grid()
plt.title('dummy data')
plt.show()