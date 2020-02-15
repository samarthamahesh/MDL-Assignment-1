import pickle
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from prettytable import PrettyTable

file = open('./Q2_data/X_train.pkl', 'rb')
x_train = pickle.load(file)
file.close()

file = open('./Q2_data/Y_train.pkl', 'rb')
y_train = pickle.load(file)
file.close()

file = open('./Q2_data/X_test.pkl', 'rb')
x_test_data = pickle.load(file)
file.close()

file = open('./Q2_data/Fx_test.pkl', 'rb')
y_test_data = pickle.load(file)
file.close()

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test_data = [[x] for x in x_test_data]

final_bias = []
final_variance = []

for degree in range(1, 10):
    bias = np.zeros(80)
    variance = np.zeros(80)
    y_predicted = []

    for set_iter in range(20):
        x_train_data = x_train[set_iter]
        y_train_data = y_train[set_iter]
        x_train_data = [[x] for x in x_train_data]

        polyFeature = PolynomialFeatures(degree)
        x_train_data = polyFeature.fit_transform(x_train_data)
        x_test_poly = polyFeature.fit_transform(x_test_data)

        reg = LinearRegression().fit(x_train_data, y_train_data)
        y_predicted = reg.predict(x_test_poly)
        bias = np.add(bias, y_predicted)
        variance = np.add(variance, y_predicted**2)

    bias /= 20                                          # E[y']
    variance /= 20                                      # E[y'^2]

    variance = np.subtract(variance, np.square(bias))   # var = E[y'^2] - E[y']^2
    bias = np.subtract(bias, y_test_data)               # bias = E[y'] - y
    bias = np.square(bias)
    
    final_bias.append(np.average(bias))
    final_variance.append(np.average(variance))

table = PrettyTable()
table.field_names = ["Degree", "Bias", "Variance"]

for i in range(9):
    table.add_row([i+1, final_bias[i], final_variance[i]])

print(table)

plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], final_bias)
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], final_variance)
plt.xlabel("Model Complexity")
plt.show()