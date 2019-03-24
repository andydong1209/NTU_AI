import pandas as pd
import pandas_datareader.data as web

# Use CSV
sp500 = pd.read_csv('GSPC_2009_2013.csv', usecols=[0, 5], index_col=0, parse_dates=True)

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
sp500.plot()
plt.show()
