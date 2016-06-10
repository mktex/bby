from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.contrib import skflow
import seaborn
import sklearn
from sklearn.preprocessing import StandardScaler
import pandas as pd

xdata = pd.read_csv('../data/toTF.csv', sep=';')
# xdata = pd.read_csv('../data/stl.csv', sep=',')
xcolumns = ['atemp', 'holiday_0', 'holiday_1', 'hr_0', 'hr_1', 'hr_10', 'hr_11', 'hr_12', 'hr_13', 'hr_14',
            'hr_15', 'hr_16', 'hr_17', 'hr_18', 'hr_19', 'hr_2', 'hr_20', 'hr_21', 'hr_22', 'hr_23', 'hr_6', 'hr_7',
            'hr_8', 'hr_9', 'hum', 'mnth_1', 'mnth_10', 'mnth_11', 'mnth_12', 'mnth_2', 'mnth_3', 'mnth_4',
            'mnth_5', 'mnth_6', 'mnth_7', 'mnth_8', 'mnth_9', 'season_1', 'season_2', 'season_3', 'season_4',
            'temp', 'weathersit_0', 'weathersit_1', 'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
            'weekday_5', 'weekday_6', 'windy', 'workingday_0', 'workingday_1', 'yr']
xl = 'cntN'
xtrain = xdata[xcolumns];
ytrain = xdata[xl]

xtrain, xtest, ytrain, ytest = sklearn.cross_validation.train_test_split(xtrain, ytrain, test_size=0.2,
                                                                         random_state=42)
# multioutput regression example - adapted
regressors = []
# options = [[2], [10, 10], [20, 20], [32, 32, 32]]
options = [[20, 20, 20]]
for hidden_units in options:
# setup exponential decay function
    global_step = 0
    def exp_decay(global_step):
        return tf.train.exponential_decay(
            learning_rate=0.1, global_step=global_step,
            decay_steps=1000, decay_rate=0.95)
    def tanh_dnn(X, y):
        features = skflow.ops.dnn(X, hidden_units=hidden_units,
                                  activation=tf.nn.tanh)
        return skflow.models.linear_regression(features, y)
    def loss_dnn(X, y):
        features = skflow.ops.dnn(X, hidden_units=hidden_units,
                                  activation=tf.nn.relu6)
        return skflow.models.linear_regression(features, y)
    regressor = skflow.TensorFlowEstimator(model_fn=loss_dnn, n_classes=0, steps=10000, learning_rate=exp_decay,
                                           batch_size=50, optimizer='Adagrad', verbose=0)
    print("[x] ..")
    regressor.fit(xtrain, ytrain)
    # http://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
    score_rmse = mean_squared_error(regressor.predict(xtest), ytest) ** 0.5
    score_mae = sklearn.metrics.mean_absolute_error(regressor.predict(xtest), ytest)
    score_explained_var = sklearn.metrics.explained_variance_score(regressor.predict(xtest), ytest)
    score_r2 = sklearn.metrics.r2_score(regressor.predict(xtest), ytest)
    print("Root Mean Squared Error for {0}: {1:f}".format(str(hidden_units), score_rmse))
    print("Mean Absolute Error for {0}: {1:f}".format(str(hidden_units), score_mae))
    print("Explained Variance for {0}: {1:f}".format(str(hidden_units), score_explained_var))
    print("R2 for {0}: {1:f}".format(str(hidden_units), score_r2))
    regressors.append(regressor)
    # skflow.TensorFlowDNNRegressor

# A. learning_rate 0.1: 0.01595, 0.01245, 0.01467 :: 0.030404, 0.024292, 0.026859
# B. learning rate with exponential decay: 0.01570, 0.01223, 0.01096 :: 0.029285, 0.022817, 0.019145
# C. learning rate with exponential decay, model loss_dnn: 0.01245, 0.00986, 0.00865 :: 0.023117, 0.018076, 0.016518

# Plot the results
y_1 = regressors[0].predict(xtest)
# y_2 = regressors[1].predict(xtest)
# y_3 = regressors[2].predict(xtest)
# y_4 = regressors[3].predict(xtest)
plt.figure()
plt.scatter(ytest, ytest, c="g", label="data")
plt.scatter(ytest, y_1[:, 0], c="y", label="hidden_units{}".format(str(options[0])))
# plt.scatter(ytest, y_2[:, 0], c="r", label="hidden_units{}".format(str(options[1])))
# plt.scatter(ytest, y_3[:, 0], c="b", label="hidden_units{}".format(str(options[2])))
# plt.scatter(ytest, y_4[:, 0], c="c", label="hidden_units{}".format(str(options[3])))
# plt.xlim([-6, 6])
# plt.ylim([-6, 6])
plt.xlabel("data")
plt.ylabel("target")
plt.title("DNN Regression")
plt.legend()
plt.show()


'''
xdata.stl <- stl(ts(xdata$cntN, frequency=360), 'periodic')
xdata$seasonal <- xdata.stl$time.series[, 1]
xdata$trend <- xdata.stl$time.series[, 2]
xdata$remainder <- xdata.stl$time.series[, 3]
write.csv(xdata, './data/stl.csv')

reading
http://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
'''