import pandas as pd
import pandas_datareader.data as web

google = web.DataReader('GOOG', data_source='yahoo', start='1/1/2004', end='12/31/2016')
print(google.info())
print(google.head())

"""
# use IEX
# data must within 5 years past
start='3/3/2018'
end='3/6/2019'
google = web.DataReader("GOOG", "iex", start, end)

# Use quandl
google = data.DataReader("GOOG", 'quandl', start, end, access_key = "YOUR API KEY FROM QUANDL")

# Use CSV

google = pd.read_csv('GOOGL.csv', usecols=[0, 4, 5, 6], index_col=0, parse_dates=True)
google = pd.read_csv('GOOGL.csv', usecols=['Date', 'Close', 'Adj Close'], index_col=0, parse_dates=True)

"""

google = google['Close']

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
google.plot()
plt.show()


google.plot(alpha=0.5, style='-')
google.resample('BA').mean().plot(style=':')
google.asfreq('BA').plot(style='--');
plt.legend(['input', 'resample', 'asfreq'], loc='upper left')
plt.show()


fig, ax = plt.subplots(3, sharey=True)
# apply a frequency to the data
google = google.asfreq('D', method='pad')
google.plot(ax=ax[0])
google.shift(900).plot(ax=ax[1])
google.tshift(900).plot(ax=ax[2])
# legends and annotations
local_max = pd.to_datetime('2007-11-05')
offset = pd.Timedelta(900, 'D')
ax[0].legend(['input'], loc=2)
ax[0].get_xticklabels()[4].set(weight='heavy', color='red')
ax[0].axvline(local_max, alpha=0.3, color='red')
ax[1].legend(['shift(900)'], loc=2)
ax[1].get_xticklabels()[4].set(weight='heavy', color='red')
ax[1].axvline(local_max + offset, alpha=0.3, color='red')
ax[2].legend(['tshift(900)'], loc=2)
ax[2].get_xticklabels()[1].set(weight='heavy', color='red')
ax[2].axvline(local_max + offset, alpha=0.3, color='red')
plt.show()


ROI = 100 * (google.tshift(-365) / google - 1)
ROI.plot()
plt.ylabel('% Return on Investment');
plt.show()


rolling = google.rolling(365, center=True)
data = pd.DataFrame({'input': google, 'one-year rolling_mean': rolling.mean(), 
	'one-year rolling_std': rolling.std()})
ax = data.plot(style=['-', '--', ':'])
ax.lines[0].set_alpha(0.3)
