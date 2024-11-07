import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Modelo de regresion lineal que se aplica a los datos de valor de la plantilla frente 
# al rendimiento de una plantilla.
def investment_performance_squad(data):
    x = data['Squad Value']
    y = data['Points']

    x = x.values.reshape(-1, 1)
    y = y.values.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101)
    model = LinearRegression()

    model.fit(x_train, y_train)
    print(model.coef_)

    predictions = model.predict(x_test)
    plt.scatter(x_test, y_test, color = 'blue')
    plt.plot(x_test, predictions, color = 'red', linewidth = 3)
    plt.title("Squad value vs points - Model One")

    plt.title('How many points out is each prediction?')
    sns.distplot((y_test-predictions),bins=50, color = 'purple')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))

