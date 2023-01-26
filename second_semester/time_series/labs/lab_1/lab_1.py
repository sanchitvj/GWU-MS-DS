import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("tute1.csv", header=0,
                   parse_dates=[0],
                   index_col=0)

print(df.head())

# df['Date'] = pd.to_datetime(df['Date'])
#
# df.set_index('Date', inplace=True)

df.plot(kind='line')
# plt.plot(df['Sales'], label='Sales')
# plt.plot(df['AdBudget'], label='AdBudget')
# plt.plot(df['GDP'], label='GDP')

# plt.figure(figsize=(10,10))
plt.xlabel('Date')
plt.ylabel('USD($)')
plt.grid()
plt.legend(loc='upper left')

plt.show()

print(f"The Sales mean is: {df.Sales.mean():.2f} and the variance is: {df.Sales.var():.2f} "
      f"with standard deviation: {df.Sales.std():.2f} medain: {df.Sales.median():.2f}")
print(f"The AdBudget mean is: {df.AdBudget.mean():.2f} and the variance is: {df.AdBudget.var():.2f} "
      f"with standard deviation: {df.AdBudget.std():.2f} medain: {df.AdBudget.median():.2f}")
print(f"The GDP mean is: {df.GDP.mean():.2f} and the variance is: {df.GDP.var():.2f} "
      f"with standard deviation: {df.GDP.std():.2f} medain: {df.GDP .median():.2f}")
