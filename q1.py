import pickle
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

file = open('Q1_data/data.pkl', 'rb')
data = pickle.load(file)
file.close()

train, test = train_test_split(data, test_size=0.1)

train = np.array(train)
train = np.split(train, 10)
train = np.array(train)
test = np.array(test)
x_test = test[:, 0]
y_test = test[:, 1]
x_test = [[x] for x in x_test]

mean_bias = []

for degree in range(1, 10):
    bias = np.zeros(500)
    y_predicted = []

    for set_iter in range(10):
        x_train = train[set_iter][:, 0]
        y_train = train[set_iter][:, 1]
        x_train = [[x] for x in x_train]

        polyFeature = PolynomialFeatures(degree)
        x_train = polyFeature.fit_transform(x_train)
        x_test_poly = polyFeature.fit_transform(x_test)
        reg = LinearRegression().fit(x_train, y_train)
        y_predicted = reg.predict(x_test_poly)
        bias += y_predicted

    bias /= 10
    bias = np.subtract(bias, y_test)

    mean_bias.append(np.average(bias))

mean_bias = np.array(mean_bias)
mean_bias = np.square(mean_bias)
print(mean_bias)