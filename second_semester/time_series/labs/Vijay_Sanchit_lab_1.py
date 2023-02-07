import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolbox import cal_rolling_mean_var, adf_test, kpss_test, non_seasonal_differencing
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("tute1.csv", header=0,
                 parse_dates=[0],
                 index_col=0)

# print(df.head())
df.plot(kind='line')
plt.xlabel('Date')
plt.ylabel('USD($)')
plt.grid()
plt.legend(loc='upper left')
plt.subplots_adjust(bottom=0.15)
plt.show()

print(f"The Sales mean is: {df.Sales.mean():.2f} and the variance is: {df.Sales.var():.2f} "
      f"with standard deviation: {df.Sales.std():.2f} median: {df.Sales.median():.2f}")
print(f"The AdBudget mean is: {df.AdBudget.mean():.2f} and the variance is: {df.AdBudget.var():.2f} "
      f"with standard deviation: {df.AdBudget.std():.2f} median: {df.AdBudget.median():.2f}")
print(f"The GDP mean is: {df.GDP.mean():.2f} and the variance is: {df.GDP.var():.2f} "
      f"with standard deviation: {df.GDP.std():.2f} median: {df.GDP.median():.2f}")

# print(df['Sales'].rolling(100).mean())
# print(df.Sales[0:1].mean())

rolling_mean_sales, rolling_var_sales = cal_rolling_mean_var(df, "Sales")
rolling_mean_adbudget, rolling_var_adbudget = cal_rolling_mean_var(df, "AdBudget")
rolling_mean_gdp, rolling_var_gdp = cal_rolling_mean_var(df, "GDP")

plt.subplot(2, 1, 1)
plt.plot(rolling_mean_sales, 'b', color='cyan')
plt.title('Rolling Mean - Sales')
plt.xlabel('Samples')
plt.ylabel('Magnitude')

plt.subplot(2, 1, 2)
plt.plot(rolling_var_sales, 'b', label="Varying variance", color='cyan')
plt.title('Rolling Variance - Sales')
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.legend()

plt.subplots_adjust(bottom=0.15)
plt.tight_layout(h_pad=2.2, w_pad=2)
plt.show()


plt.subplot(2, 1, 1)
plt.plot(rolling_mean_adbudget, 'b', color='violet')
plt.title('Rolling Mean - AdBudget')
plt.xlabel('Samples')
plt.ylabel('Magnitude')

plt.subplot(2, 1, 2)
plt.plot(rolling_var_adbudget, 'b', label="Varying variance", color='violet')
plt.title('Rolling Variance - AdBudget')
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.legend()

plt.subplots_adjust(bottom=0.15)
plt.tight_layout(h_pad=2.2, w_pad=2)
plt.show()


plt.subplot(2, 1, 1)
plt.plot(rolling_mean_gdp, 'b', color='green')
plt.title('Rolling Mean - GDP')
plt.xlabel('Samples')
plt.ylabel('Magnitude')

plt.subplot(2, 1, 2)
plt.plot(rolling_var_gdp, 'b', label="Varying variance", color='green')
plt.title('Rolling Variance - GDP')
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.legend()

plt.subplots_adjust(bottom=0.15)
plt.tight_layout(h_pad=2.2, w_pad=2)
plt.show()

print("ADF test for Sales:")
adf_test(df.Sales)
print("ADF test for AdBudget:")
adf_test(df.AdBudget)
print("ADF test for GDP:")
adf_test(df.GDP)

print("KPSS test for Sales:")
kpss_test(df.Sales)
print("KPSS test for AdBudget:")
kpss_test(df.AdBudget)
print("KPSS test for GDP:")
kpss_test(df.GDP)

df_ap = pd.read_csv("AirPassengers.csv", header=0,
                 parse_dates=[0],
                 index_col=0)
df_ap.rename(columns={"#Passengers": "passengers"}, inplace=True)
# print(df_ap.head())

df_ap.plot(kind='line')
plt.xlabel('Date')
plt.ylabel('No. of Passengers')
plt.grid()
plt.legend(loc='upper left')
plt.subplots_adjust(bottom=0.15)
plt.show()

print(f"The no. passengers mean is: {df_ap.passengers.mean():.2f} and the variance is: {df_ap.passengers.var():.2f} "
      f"with standard deviation: {df_ap.passengers.std():.2f} median: {df_ap.passengers.median():.2f}")

rolling_mean_pass, rolling_var_pass = cal_rolling_mean_var(df_ap, "passengers")

plt.subplot(2, 1, 1)
plt.plot(rolling_mean_pass, 'b', color='red')
plt.title('Rolling Mean - #Passengers')
plt.xlabel('Samples')
plt.ylabel('Magnitude')

plt.subplot(2, 1, 2)
plt.plot(rolling_var_pass, 'b', label="Varying variance", color='red')
plt.title('Rolling Variance - #Passengers')
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.legend()

plt.subplots_adjust(bottom=0.15)
plt.tight_layout(h_pad=2.2, w_pad=2)
plt.show()

print("ADF test for #Passengers:")
adf_test(df_ap.passengers)

print("KPSS test for #Passengers:")
kpss_test(df_ap.passengers)


df_ap["first_diff"] = non_seasonal_differencing(df_ap, "passengers", 1)
rolling_mean_first, rolling_var_first = cal_rolling_mean_var(df_ap, "first_diff")

plt.subplot(2, 1, 1)
plt.plot(rolling_mean_first, 'b', color='red')
plt.title('Rolling Mean - 1st order difference')
plt.xlabel('Samples')
plt.ylabel('Magnitude')

plt.subplot(2, 1, 2)
plt.plot(rolling_var_first, 'b', label="Varying variance", color='red')
plt.title('Rolling Variance - 1st order difference')
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.legend()

plt.subplots_adjust(bottom=0.15)
plt.tight_layout(h_pad=2.2, w_pad=2)
plt.show()

print("ADF test for 1st order difference:")
adf_test(df_ap.first_diff)

print("KPSS test for 1st order difference:")
kpss_test(df_ap.first_diff)


df_ap["second_diff"] = non_seasonal_differencing(df_ap, "first_diff", 2)
rolling_mean_second, rolling_var_second = cal_rolling_mean_var(df_ap, "second_diff")

plt.subplot(2, 1, 1)
plt.plot(rolling_mean_second, 'b', color='red')
plt.title('Rolling Mean - 2nd order difference')
plt.xlabel('Samples')
plt.ylabel('Magnitude')

plt.subplot(2, 1, 2)
plt.plot(rolling_var_second, 'b', label="Varying variance", color='red')
plt.title('Rolling Variance - 2nd order difference')
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.legend()

plt.subplots_adjust(bottom=0.15)
plt.tight_layout(h_pad=2.2, w_pad=2)
plt.show()

print("ADF test for 2nd order difference:")
adf_test(df_ap.second_diff)

print("KPSS test for 2nd order difference:")
kpss_test(df_ap.second_diff)


df_ap["third_diff"] = non_seasonal_differencing(df_ap, "second_diff", 3)
rolling_mean_third, rolling_var_third = cal_rolling_mean_var(df_ap, "third_diff")

plt.subplot(2, 1, 1)
plt.plot(rolling_mean_third, 'b', color='red')
plt.title('Rolling Mean - 3rd order difference')
plt.xlabel('Samples')
plt.ylabel('Magnitude')

plt.subplot(2, 1, 2)
plt.plot(rolling_var_third, 'b', label="Varying variance", color='red')
plt.title('Rolling Variance - 3rd order difference')
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.legend()

plt.subplots_adjust(bottom=0.15)
plt.tight_layout(h_pad=2.2, w_pad=2)
plt.show()

print("ADF test for 3rd order difference:")
adf_test(df_ap.third_diff)

print("KPSS test for 3rd order difference:")
kpss_test(df_ap.third_diff)


df_ap["log_transformed"] = np.log(df_ap.passengers)

df_ap["log_first_diff"] = non_seasonal_differencing(df_ap, "log_transformed", 1)
rolling_mean_log_first, rolling_var_log_first = cal_rolling_mean_var(df_ap, "log_first_diff")

plt.subplot(2, 1, 1)
plt.plot(rolling_mean_log_first, 'b', color='red')
plt.title('Rolling Mean - log transformed 1st order difference')
plt.xlabel('Samples')
plt.ylabel('Magnitude')

plt.subplot(2, 1, 2)
plt.plot(rolling_var_log_first, 'b', label="Varying variance", color='red')
plt.title('Rolling Variance - log transformed 1st order difference')
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.legend()

plt.subplots_adjust(bottom=0.15)
plt.tight_layout(h_pad=2.2, w_pad=2)
plt.show()

print("ADF test for log transformed 1st order difference:")
adf_test(df_ap.log_first_diff)

print("KPSS test for log transformed 1st order difference:")
kpss_test(df_ap.log_first_diff)