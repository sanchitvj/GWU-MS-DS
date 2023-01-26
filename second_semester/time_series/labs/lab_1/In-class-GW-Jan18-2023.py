import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('CO2_1970-2015_dataset_of_CO2_report_2016.xls',
                   header=0,
                   parse_dates=[0],
                   index_col=0)
Year = np.arange(1970,2016)
CO2_In = df.loc['Indonesia']
CO2_Sw = df.loc['Sweden']

plt.figure()
plt.plot(Year, CO2_In, label = 'Indonesia')
plt.plot(Year, CO2_Sw, label = 'Sweden')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Co2 Emission Level')
plt.title('Co2 Emission from fossile use per country')
plt.tight_layout()
plt.show()

Euro = ['Estonia','France', 'Germany', 'Hungary', 'Italy', 'Russian Federation', 'Spain', 'Sweden','United Kingdom']
CO2 = df.loc[Euro,:]
CO2.loc['year']= Year
CO2.T.plot(x= 'year', legend = None)
plt.legend(Euro)
plt.xlabel('Year')
plt.ylabel('Co2 Emission Level')
plt.title('Co2 Emission from fossile use per European country')
plt.show()
ASEAN = ['Brunei Darussalam','Cambodia', 'Lao People s Democratic Republic', 'Malaysia', 'Myanmar', 'Singapore', 'Thailand', 'Viet Nam']
CO2 = df.loc[ASEAN,:]
CO2.loc['year']= Year
CO2.T.plot(x= 'year', legend = None)
plt.legend(ASEAN)
plt.xlabel('Year')
plt.ylabel('Co2 Emission Level')
plt.title('Co2 Emission from fossile use per ASEAN country')
plt.tight_layout()
plt.show()