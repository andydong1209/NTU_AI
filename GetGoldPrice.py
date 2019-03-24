import pandas as pd
import matplotlib.pyplot as plt

gold = pd.read_csv('Gold.csv', usecols=[0, 3], index_col=0, parse_dates=True)

gold.plot()
plt.show()


