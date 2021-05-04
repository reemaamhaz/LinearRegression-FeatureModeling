import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.optimize import minimize
import sys


data = sns.load_dataset("tips")

print("Number of Records:", len(data))
data.head()

tips = data['tip']
X = data.drop(columns='tip')

def one_hot_encode(data):
    return pd.get_dummies(data) # fill the columns with 0's and 1

one_hot_X = one_hot_encode(X)
one_hot_X.head()


def linear_model(thetas, X):
    return np.dot(X, thetas) #get the product of the matrices 


def mae(y, y_hat):
    return np.abs(np.mean(y_hat) - np.mean(y)) # get the absolute value of the difference in means




def mse(y, y_hat):
    return np.abs(np.mean(y) - np.mean(y_hat))**2 # get the absolute value of the difference in means squared


def make_function_to_minimize(average_loss_function, model, X, y):
    def my_function_to_minimize(theta):
        return average_loss_function(model(theta, X), y)
    
    return my_function_to_minimize



mse_function_to_minimize = make_function_to_minimize(mse, linear_model, one_hot_X, data['tip'])

mse_thetas = minimize(mse_function_to_minimize, x0=np.zeros(one_hot_X.shape[1]))['x']



print("The values for theta that minimize Mean Square Error are: \n\n ", mse_thetas)
print("\n The value of the Mean Square Error is: \n\n", mse(linear_model(mse_thetas, one_hot_X), tips))


mae_function_to_minimize = make_function_to_minimize(mae, linear_model, one_hot_X, data['tip'])

mae_thetas = minimize(mae_function_to_minimize, x0=np.zeros(one_hot_X.shape[1]))['x']


print("The values for theta that minimize Mean Square Error are: \n\n ", mae_thetas)
print("\n The value of the Mean Square Error is: \n\n", mae(linear_model(mae_thetas, one_hot_X), tips))

