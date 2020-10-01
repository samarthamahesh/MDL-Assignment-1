import pickle
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from prettytable import PrettyTable

file = open('Q1_data/data.pkl', 'rb')
data = pickle.load(file)
file.close()

train, test = train_test_split(data, test_size=0.1)

train = np.array(train)
train = np.split(train, 10)
train = np.array(train)
x_test = test[:, 0]
y_test = test[:, 1]
x_test = [[x] for x in x_test]

final_bias = []
final_variance = []

for degree in range(1, 10):
    bias = np.zeros(500)
    variance = np.zeros(500)
    y_predicted = []

    for set_iter in range(10):
        x_train = train[set_iter][:, 0]
        y_train_data = train[set_iter][:, 1]
        x_train = [[x] for x in x_train]

        polyFeature = PolynomialFeatures(degree)
        x_train_data = polyFeature.fit_transform(x_train)
        x_test_poly = polyFeature.fit_transform(x_test)

        reg = LinearRegression().fit(x_train_data, y_train_data)
        y_predicted = reg.predict(x_test_poly)
        bias = np.add(bias, y_predicted)
        variance = np.add(variance, y_predicted**2)

    bias /= 10
    variance /= 10

    variance = np.subtract(variance, np.square(bias))
    bias = np.subtract(bias, y_test)
    bias = np.square(bias)

    final_bias.append(np.average(bias))
    final_variance.append(np.average(variance))

table = PrettyTable()
table.field_names = ["Degree", "Bias", "Variance"]

for i in range(9):
    table.add_row([i+1, final_bias[i], final_variance[i]])

print(table)
