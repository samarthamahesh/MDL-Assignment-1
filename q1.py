import pickle
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def return_coef(train_set, degree):
    eq = []
    x_train = train_set[:, 0]
    y_train = train_set[:, 1]
    for i in x_train:
        temp = []
        for j in range(1, degree+1):
            temp.append(i**j)
        eq.append(temp)

    reg.fit(eq, y_train)
    coef = reg.coef_
    return coef

def calculate_y(x, coef, degree):
    y = 0
    for i in range(1, degree+1):
        y += coef[i-1]*(x**i)
    return y

reg = linear_model.LinearRegression()

file = open('../Q1_data/data.pkl', 'rb')
data = pickle.load(file)
file.close()


train, test = train_test_split(data, test_size=0.1)

split_train = np.split(train, 10)

X_test = test[:, 0]
Y_test = test[:, 1]

biases = []
### Bias of classifiers / models
for degree in range(1, 10):
    models = []
    for iter in range(10):
        coef = return_coef(split_train[iter], degree)
        models.append(coef)

    y_sum = 0

    for iter in range(10):
        y_predict = calculate_y(X_test, models[iter], degree)
        y_sum += y_predict

    y_mean = y_sum / 10

    bias = y_mean - Y_test

    sum = np.sum(bias)
    mean_bias = sum / len(y_mean)
    biases.append(mean_bias)

print(biases)

variances = []
### Variance of classifiers / models
# for degree in range(1, 10):
#     models = []
#     for iter in range(10):
#         coef = return_coef(split_train[iter], degree)
#         models.append(coef)

#     y_sum = 0

#     y_predicted_col = []

#     for iter in range(10):
#         y_predicted = calculate_y(X_test, models[iter], degree)
#         y_sum += y_predicted
#         y_predicted_col.append(y_predicted)

#     y_mean = y_sum / 10

#     for iter in range(len(y_predicted_col)):
#         y_predicted_col[iter] -= y_mean

#     expected_squared = np.square(y_predicted_col)

#     variance = 0

#     length = len(expected_squared)
#     for iter in range(length):
#         variance += expected_squared[iter]

#     length = len(variance)
#     sum = np.sum(variance)

#     mean_variance = sum / length
#     variances.append(mean_variance)

# # print(biases)
# # print(variances)

# bias_plot = np.square(biases)
# variance_plot = variances

# print(bias_plot)
# print(variance_plot)

# # plt.plot(bias_plot)
# # plt.plot(variance_plot)
# # plt.show()