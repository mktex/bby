# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import tensorflow as tf
import urllib
import zipfile
import os
import scipy.stats as stats


pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 2)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/'
filesize = 279992
filename = "Bike-Sharing-Dataset.zip"

# Source: tensorflow examples - word2vec_basic.py
def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

def load_data():
    maybe_download(filename, filesize)
    xF = zipfile.ZipFile(filename)
    xF.extractall('../data/')
    xF.close()

# Outliers detection
def isOutlier(xSeries):
    try:
        int(xSeries.iloc[0])
    except:
        return map(lambda x: np.nan, xSeries)
    x25 = xSeries.describe()['25%']
    x75 = xSeries.describe()['75%']
    xmean = xSeries.mean()
    return map(lambda x: x<(xmean - 1.5*x25) or x>(xmean + 1.5*x75), xSeries)

def compTTest(mu1, sd1, n1, mu2, sd2, n2):
    from scipy import stats
    np.random.seed(12345678)
    rvs1 = stats.norm.rvs(loc=mu1,scale=sd1,size=n1)
    rvs2 = stats.norm.rvs(loc=mu2,scale=sd2,size=n2)
    stats.ttest_ind(rvs1,rvs2)
    return stats.ttest_ind(rvs1,rvs2, equal_var = False)

# x if np.isnan(x) else
def normalization(xSeries):
    xmin = xSeries.min()
    xmax = xSeries.max()
    return map(lambda x: round(((x - xmin)/(xmax - xmin)), 3), xSeries)

# x if np.isnan(x) else
def standardize(xSeries):
    xmean = xSeries.mean()
    xstd = xSeries.std(ddof=0)
    return map(lambda x: round(((x - xmean)/xstd), 3), xSeries)

import scipy.stats as stats
def calculate_skewness(x):
    m1 = np.mean(x);
    m2 = np.median(x);
    m3 = np.std(x);
    return (3 * (m1 - m2)) / m3

def transform_skewness_to_normal(xSeries):
    return map(lambda x: 1.0 / np.sqrt(x), xSeries)

def turnValFromNormalizedBack(xdata, xval=50):
    xgroup = xdata
    # xgroup['FareAdj'][:10]
    # xgroup['FareAdjN'][:10]
    # xgroup['FareAdjN'][41]
    # xval = xgroup['FareAdjN'][41]
    # xval_1 = 2.493
    # xval_2 = 20.8
    # xval = 2.293
    xres = 1. / np.power(xval, 2)
    xres = xres - 0.1
    xmean = xgroup.mean()
    xstd = xgroup.std(ddof=0)
    xstandardized = pd.Series(standardize(xgroup))
    xmin = xstandardized.min()
    xmax = xstandardized.max()
    xres = xres * (xmax - xmin) + xmin
    xres = xres * xstd + xmean
    return xres


# =========================   DATA LOADING

load_data()

# if all went ok, then I'm ready to load data in pandas.
xd1 = pd.read_csv('../data/day.csv')
xd2 = pd.read_csv('../data/hour.csv')

# Describe
xd1.describe()
xd2.describe()


# =========================   NULL, NA - values
# do we have any null values..?
for xcol in xd2.columns:
    print xcol, len(xd2[xd2[xcol].isnull()])
print "[x] null values?"

# =========================   data prep, wrong values
# statistical significant differences between sample populations of groups by season
# are there statistically different groups? use t-test
# is cnt for each season normaly distributed? yes!
stats.normaltest(xd2[xd2['season']==1]['cnt'])
# NormaltestResult(statistic=1637.1915212328008, pvalue=0.0)
print stats.normaltest(xd2[xd2['season']==2]['cnt'])
# NormaltestResult(statistic=589.09932909694351, pvalue=1.1986876756714587e-128)
print stats.normaltest(xd2[xd2['season']==3]['cnt'])
# NormaltestResult(statistic=572.77495474929265, pvalue=4.202414535924059e-125)
print stats.normaltest(xd2[xd2['season']==4]['cnt'])
# NormaltestResult(statistic=822.26640741966514, pvalue=2.7997438646932384e-179)
x = xd2.groupby('season')[['cnt', 'casual', 'registered']].mean()
y = xd2.groupby('season')[['cnt', 'casual', 'registered']].std()
z = xd2.groupby('season')[['cnt', 'casual', 'registered']].count()
print x
print y
print z
xcomp = [(0, 1), (1, 2), (2, 3), (0, 3)]
for a, b in xcomp:
    print "\n cnt for season %d vs %d"%(a+1, b+1)
    compTTest(mu1=x.values[a, 0], sd1=y.values[a, 0], n1=y.values[a, 0], mu2=x.values[b, 0], sd2=y.values[b, 0], n2=z.values[b, 0])

# seems we have wrong data
wrong_season = xd2[map(lambda x: (x[0] == 12 or x[0] == 1 or x[0] == 2) and x[1]!=4, zip(xd2['mnth'], xd2['season']))].index
xd2.ix[wrong_season, 'season'] = map(lambda x: 4, xrange(len(wrong_season)))
wrong_season = xd2[map(lambda x: (x[0] == 3 or x[0] == 4 or x[0] == 5) and x[1]!=1, zip(xd2['mnth'], xd2['season']))].index
xd2.ix[wrong_season, 'season'] = map(lambda x: 1, xrange(len(wrong_season)))
wrong_season = xd2[map(lambda x: (x[0] == 6 or x[0] == 7 or x[0] == 8 or x[0] == 9) and x[1]!=2, zip(xd2['mnth'], xd2['season']))].index
xd2.ix[wrong_season, 'season'] = map(lambda x: 2, xrange(len(wrong_season)))
wrong_season = xd2[map(lambda x: (x[0] == 10 or x[0] == 11) and x[1]!=3, zip(xd2['mnth'], xd2['season']))].index
xd2.ix[wrong_season, 'season'] = map(lambda x: 3, xrange(len(wrong_season)))

# this should be done for xd1 as well, in case I will work with it later..
wrong_season = xd1[map(lambda x: (x[0] == 12 or x[0] == 1 or x[0] == 2) and x[1]!=4, zip(xd1['mnth'], xd1['season']))].index
xd1.ix[wrong_season, 'season'] = map(lambda x: 4, xrange(len(wrong_season)))
wrong_season = xd1[map(lambda x: (x[0] == 3 or x[0] == 4 or x[0] == 5) and x[1]!=1, zip(xd1['mnth'], xd1['season']))].index
xd1.ix[wrong_season, 'season'] = map(lambda x: 1, xrange(len(wrong_season)))
wrong_season = xd1[map(lambda x: (x[0] == 6 or x[0] == 7 or x[0] == 8 or x[0] == 9) and x[1]!=2, zip(xd1['mnth'], xd1['season']))].index
xd1.ix[wrong_season, 'season'] = map(lambda x: 2, xrange(len(wrong_season)))
wrong_season = xd1[map(lambda x: (x[0] == 10 or x[0] == 11) and x[1]!=3, zip(xd1['mnth'], xd1['season']))].index
xd1.ix[wrong_season, 'season'] = map(lambda x: 3, xrange(len(wrong_season)))

print "[x] finished fixing season variable"

# expect seasonality to play a role, but also a general increase in usage as the service gets wider exposed to people
xd2['yr'].describe()
xd2['mnth'].describe()
xd2['hr'].describe()

# variable holiday
xd2.groupby('holiday')[['cnt', 'season', 'mnth', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'windspeed']].describe()

# there is a problem with weathersit, for type 4 there are only 3 records and they should be removed from further analysis
xd2.groupby(['workingday', 'weathersit'])[['cnt', 'casual', 'registered']].describe()

# for temp, atemp, hum and windspeed it might be a good idea to bin data, by say 5 levels (0, 0.25, 0.5, 0.75, 1)
xd2.groupby('temp')[['cnt', 'casual', 'registered']].describe()

# casual and registered obviously contain predictive value, so the should not be used as is.
# instead we could use feature engineering and add two variables casualAvg and registeredAvg with the averages from the past N days.
# a test would be required to assess how usefull this would be

xres = xd2[xd2['weathersit']<=2].index
xd2.ix[xres, 'weathersit'] = map(lambda x: 1, xrange(len(xres)))
xres = xd2[xd2['weathersit']>2].index
xd2.ix[xres, 'weathersit'] = map(lambda x: 0, xrange(len(xres)))
# repeat it for xd1
xres = xd1[xd1['weathersit']<=2].index
xd1.ix[xres, 'weathersit'] = map(lambda x: 1, xrange(len(xres)))
xres = xd1[xd1['weathersit']>2].index
xd1.ix[xres, 'weathersit'] = map(lambda x: 0, xrange(len(xres)))

xd2.groupby('weathersit').describe()


# =========================   OUTLIERS

# no outliers from temp, atemp, hum
xd2[isOutlier(xd2['temp'])]['temp'].describe()
xd2[isOutlier(xd2['atemp'])]['atemp'].describe()
xd2[isOutlier(xd2['hum'])]['hum'].describe()
print "[x] No outliers in temp, atemp, hum.."

xd2[isOutlier(xd2['windspeed'])]['windspeed'].describe()

# windspeed has outliers, so I could be looking at extreme weather cases.
xd2[isOutlier(xd2['windspeed'])]['windspeed'].describe()
# set(xd2[isOutlier(xd2['windspeed'])]['windspeed'])
# set([0.0, 0.68659999999999999, 0.71640000000000004, 0.74629999999999996, 0.77610000000000001,
# 0.80599999999999994, 0.83579999999999999, 0.85070000000000001, 0.58209999999999995,
# 0.6119, 0.64180000000000004, 0.65669999999999995])

# there are 2180 records containing windspeed == 0. they could be an indication of
xd2[xd2['windspeed']==0]['windspeed'].count()

# for good weather, 15092 datapoints and no outliers
xr = xd2[map(lambda x: x[0] and x[1], zip( (xd2['windspeed']<0.58), (xd2['windspeed']>0) ))]
xr[isOutlier(xr['windspeed'])]

# I can check that this is bad weather (weathersit >= 1) mostly happening in spring and summer.
xd2[xd2['windspeed']>=0.58].describe()
xd2[xd2['windspeed']>=0.58].groupby('season').describe()

xd2['windy'] = map(lambda x: 1, xrange(len(xd2['windspeed'])))
xr = xd2[map(lambda x: x[0] and x[1], zip( (xd2['windspeed']<0.58), (xd2['windspeed']>0) ))].index
xd2.ix[xr, 'windy'] = map(lambda x: 0, xrange(len(xr)))
xd2.groupby('windy').describe()


# aggregated daily count data contains no outliers, but the the hourly data does
xd1[isOutlier(xd1['cnt'])]['cnt']
xd2[isOutlier(xd2['cnt'])]['cnt']
# because the number is relatively high, this could suggest there is some underlying process or phenomenon that I need to use
# in order to split the data in correct groups and learn different models on them
# The reason could be because of the fact there are high differences in usage during the day. This is logical and
# for that the bining proposed earlier for hr variable should help.

# http://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/
from pandas.tools.plotting import scatter_matrix

scatter_matrix(xd2[['windspeed', 'cnt', 'casual', 'registered']], alpha=0.2, figsize=(6, 6), diagonal='kde'); plt.show()

# xstr = [u'season', u'yr', u'mnth', u'hr', u'weathersit', 'cnt', 'registered', 'casual']
# scatter_matrix(xr[xstr], alpha=0.2, figsize=(6, 6), diagonal='hist'); plt.show()
#
# xstr = [u'holiday', u'weekday', u'workingday', 'cnt', 'registered', 'casual']
# scatter_matrix(xr[xstr], alpha=0.2, figsize=(6, 6), diagonal='kde'); plt.show()
#
# xstr = [u'temp', u'atemp', u'hum', u'windspeed', 'cnt', 'registered', 'casual']
# scatter_matrix(xr[xstr], alpha=0.2, figsize=(6, 6), diagonal='kde'); plt.show()


# normal distribution check with qqplots
#stats.probplot(xd2['temp'], dist="norm", plot=plt); plt.title('qqplot temp'); plt.show()
#stats.probplot(xd2['atemp'], dist="norm", plot=plt); plt.title('qqplot atemp'); plt.show()
#stats.probplot(xd2['hum'], dist="norm", plot=plt); plt.title('qqplot hum'); plt.show()

xstr = [u'hum', 'cnt', 'registered', 'casual']
scatter_matrix(xd2[xstr], alpha=0.2, figsize=(6, 6), diagonal='kde'); plt.show()


# the variable label itself has 8849 outliers
xd2[isOutlier(xd2['cnt'])]
xd2[isOutlier(xd2['cnt'])].describe()

# xd1 has no such outliers.
xd3 = xd2[map(lambda x: not x, isOutlier(xd2['cnt']))]
xd3['cnt'].plot(kind='box');
plt.show()

# =========================   normalisation, standardisation

# add standardisation and normalisation to labels
xd2['cntN'] = normalization(pd.Series(standardize(xd2['cnt'])))
xd2['cntN'] = map(lambda x: x + 0.1, xd2['cntN'])
xd2['cntN'] = transform_skewness_to_normal(xd2['cntN'])
xd2['cntN'].plot(kind='box');

for col in ['casual', 'registered']:
    xd2[col + 'N'] = normalization(pd.Series(standardize(xd2[col])))
    xd2[col + 'N'] = map(lambda x: x + 0.5, xd2[col])
    xd2[col + 'N'] = transform_skewness_to_normal(xd2[col])
    xd2[col + 'N'].plot(kind='box');
    plt.show()



xd1.to_csv("../data/day_res.csv");
xd2.to_csv("../data/hour_res.csv")
# just for the sake of performing an extra test
xd3.to_csv("../data/hour_res_outlier_removed_cnt.csv")



