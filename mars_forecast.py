import warnings

import numpy as np
import pandas as pd
from scipy.stats import skew

import os
import re
from datetime import datetime

from pyearth import Earth
import requests
import talib
from bs4 import BeautifulSoup
from fancyimpute import KNN
from pandas_datareader import data as web
from sklearn.ensemble import AdaBoostRegressor

warnings.filterwarnings('ignore')

# import matplotlib.pyplot as plt
# %matplotlib qt

clear = lambda: os.system('cls')


def printer(a, b, u, c, d):
    clear()
    print(a)
    print("")
    print(b)
    print("")
    print(u)
    print("")
    print(c)
    print("")
    print(d)


data = requests.get('https://finance.yahoo.com/quote/%5EVIX')
soup = BeautifulSoup(data.text, "html.parser")

vol_c = []
for span in soup.find_all('fin-streamer', {'class': "Fw(500) Pstart(8px) Fz(24px)"}):
    vol_c += span

vol_c = re.findall('\(.*?\)', str(vol_c[1]))


def main():
    asset = input('Stock Ticker: ')
    openP = input("Open Price: ")

    end = datetime.now()
    end = str(end)
    end = end[0:10]
    stock = web.DataReader(asset.upper(), 'yahoo', start=datetime(1999, 1, 1), end=end)

    train_range = int(round(0.8 * len(stock), 0))

    train = stock[0:train_range]
    test = stock[train_range:]

    train_close = train['Close']
    test_close = test['Close']

    # Preprocess: remove unnecessary columns
    cols = list(train)[0:3]
    train = train[cols].astype(str)

    train = train.astype(float)
    for i in list(train):
        if i not in cols:
            del train[i]

    ht = talib.HT_DCPERIOD(train['Open'])
    std = talib.STDDEV(train['Open'], timeperiod=14, nbdev=1)

    ht = pd.DataFrame(data={'HT_DCPERIOD': ht})
    std = pd.DataFrame(data={'STDDEV': std})

    train = train.join(ht)
    train = train.join(std)

    train = KNN(k=3).fit_transform(train)

    # transform data using log(1+x)
    train = pd.DataFrame(train)
    train[0] = np.log1p(train[0])
    y = train[0]
    z = np.log1p(train[1])
    x = pd.DataFrame(train)
    del x[0]
    del x[1]

    numeric_feats = x.dtypes[x.dtypes != "object"].index
    skewed_feats = x[numeric_feats].apply(lambda g: skew(g.dropna()))
    skewed_feats = skewed_feats.index

    x[skewed_feats] = np.log1p(x[skewed_feats])

    # Fit MARS high
    mars = Earth()
    mars.fit(x, y)
    # print(mars.trace())
    # print(mars.summary())

    # fit MARS low
    lars = Earth()
    lars.fit(x, z)

    def inverse(tnf):
        tnf = np.exp(tnf) - 1
        return tnf

    # Process test data
    for i in list(test):
        if i not in cols:
            del test[i]
            test = test.astype(float)

    test.loc['Today'] = [None, None, float(openP)]

    ht = talib.HT_DCPERIOD(test['Open'])
    std = talib.STDDEV(test['Open'], timeperiod=14, nbdev=1)

    ht = pd.DataFrame(data={'HT_DCPERIOD': ht})
    std = pd.DataFrame(data={'STDDEV': std})

    test = test.join(ht)
    test = test.join(std)

    test = KNN(k=3).fit_transform(test)
    test = pd.DataFrame(test)

    test[0] = np.log1p(test[0])
    y1 = test[0]
    z1 = np.log1p(test[1])
    x1 = test
    del x1[0]
    del x1[1]

    features = x1.dtypes[x1.dtypes != "object"].index
    features_skewed = test[features].apply(lambda g: skew(g.dropna()))
    features_skewed = features_skewed.index
    x1[features_skewed] = np.log1p(x1[features_skewed])

    # Adaboost MARS
    boosted_mars = AdaBoostRegressor(base_estimator=mars, n_estimators=25, learning_rate=0.1, loss="exponential")
    boosted_mars.fit(x, y)

    boosted_lars = AdaBoostRegressor(base_estimator=lars, n_estimators=25, learning_rate=0.1, loss="exponential")
    boosted_lars.fit(x, z)

    # Predict using boosted MARS
    # yb = boosted_mars.predict(x)
    y_hat1 = boosted_mars.predict(x1)
    z_hat1 = boosted_lars.predict(x1)

    cars = Earth()
    boosted_cars = AdaBoostRegressor(base_estimator=cars, n_estimators=25, learning_rate=0.1, loss="exponential")
    boosted_cars.fit(x, np.log1p(train_close))

    c_hat1 = boosted_cars.predict(x1)

    ##################### Accuracy HIGH #######################
    S1 = pd.DataFrame(inverse(y1))
    S1['com'] = S1[0].shift(1)
    S1['ACC'] = S1[0] - S1['com']
    S1['ACC'] = S1['ACC'].mask(S1['ACC'] > 0, 1)
    S1['ACC'] = S1['ACC'].mask(S1['ACC'] < 0, 0)

    S2 = pd.DataFrame(inverse(y_hat1))
    S2['com'] = S2[0].shift(1)
    S2['ACC2'] = S2[0] - S2['com']
    S2['ACC2'] = S2['ACC2'].mask(S2['ACC2'] > 0, 1)
    S2['ACC2'] = S2['ACC2'].mask(S2['ACC2'] < 0, 0)

    S3 = pd.DataFrame(S1['ACC'])
    S3 = S3.join(S2['ACC2'])
    S3['score'] = 0
    S3['score'] = S3['score'].mask(S3['ACC'] == S3['ACC2'], 1)

    ac = S3['score'].value_counts()
    acc1 = round((ac[1] / len(y1)) * 100, 4)
    ##########################################################

    #################### Accuracy LOW ########################
    t1 = pd.DataFrame(inverse(z1))
    t1['com'] = t1[1].shift(1)
    t1['ACC'] = t1[1] - t1['com']
    t1['ACC'] = t1['ACC'].mask(t1['ACC'] > 0, 1)
    t1['ACC'] = t1['ACC'].mask(t1['ACC'] < 0, 0)

    t2 = pd.DataFrame(inverse(z_hat1))
    t2['com'] = t2[0].shift(1)
    t2['ACC2'] = t2[0] - t2['com']
    t2['ACC2'] = t2['ACC2'].mask(t2['ACC2'] > 0, 1)
    t2['ACC2'] = t2['ACC2'].mask(t2['ACC2'] < 0, 0)

    t3 = pd.DataFrame(t1['ACC'])
    t3 = t3.join(t2['ACC2'])
    t3['score'] = 0
    t3['score'] = t3['score'].mask(t3['ACC'] == t3['ACC2'], 1)

    a_ct = t3['score'].value_counts()
    ac_ct = round((a_ct[1] / len(z1)) * 100, 4)
    ##########################################################

    #################### Accuracy CLOSE ########################
    k1 = pd.DataFrame(np.array(test_close))
    k1['com'] = k1[0].shift(1)
    k1['ACC'] = k1[0] - k1['com']
    k1['ACC'] = k1['ACC'].mask(k1['ACC'] > 0, 1)
    k1['ACC'] = k1['ACC'].mask(k1['ACC'] < 0, 0)

    k2 = pd.DataFrame(inverse(c_hat1))
    k2['com'] = k2[0].shift(1)
    k2['ACC2'] = k2[0] - k2['com']
    k2['ACC2'] = k2['ACC2'].mask(k2['ACC2'] > 0, 1)
    k2['ACC2'] = k2['ACC2'].mask(k2['ACC2'] < 0, 0)

    k3 = pd.DataFrame(k1['ACC'])
    k3 = k3.join(k2['ACC2'])
    k3['score'] = 0
    k3['score'] = k3['score'].mask(k3['ACC'] == k3['ACC2'], 1)

    a_ck = k3['score'].value_counts()
    ac_ck = round((a_ck[1] / len(test_close)) * 100, 4)
    ##########################################################

    c = 'Out of sample accuracy: ' + 'High = ' + str(acc1) + '%'', Low = ' + str(ac_ct) + '%' + ', Close = ' + str(
        ac_ck) + '%'

    price = inverse(y_hat1[-1])
    price_z = inverse(z_hat1[-1])
    price_c = inverse(c_hat1[-1])
    # y1 = np.array(y1)

    if inverse(y_hat1[-1]) > inverse(y_hat1[-2]):
        signal = 'UP'
    else:
        signal = 'DOWN'

    if inverse(z_hat1[-1]) > inverse(z_hat1[-2]):
        signalz = 'UP'
    else:
        signalz = 'DOWN'

    if inverse(c_hat1[-1]) > inverse(c_hat1[-2]):
        signalc = 'UP'
    else:
        signalc = 'DOWN'

    a = 'High Prediction: $' + str(round(price, 2)) + ' ... Direction: ' + signal
    b = 'Low Prediction: $' + str(round(price_z, 2)) + ' ... Direction: ' + signalz
    u = 'Close Prediction: $' + str(round(price_c, 2)) + ' ... Direction: ' + signalc

    v = 'VIX change: '
    d = None
    if '+' in vol_c[1]:
        d = v + vol_c[1] + ' Short Bias'
    if '-' in vol_c[1]:
        d = v + vol_c[1] + ' Long Bias'
    if vol_c[1] == '(0.00%)':
        d = v + vol_c[1] + ' No Bias'

    return printer(a, b, u, c, d)


if __name__ == '__main__':
    main()
