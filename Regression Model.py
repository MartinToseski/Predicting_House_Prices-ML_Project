import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv(
    'Databases/Everything/DB-Harlfoxem.csv')

train_data['date'] = pd.to_numeric(
    train_data['date'].str.replace('T000000', ''))

X = train_data.drop('price', axis=1)
Y = train_data['price']

model = LinearRegression()

model.fit(X, Y)

test_data = pd.read_csv('test_data.csv')

test_data['date'] = pd.to_numeric(test_data['date'].str.replace('T000000', ''))

predictions = model.predict(test_data)

print(predictions)
