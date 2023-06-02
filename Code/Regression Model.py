import pandas as pd
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit 
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

def evaluation(y, predictions):
    r_squared = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    return r_squared, mae, rmse

train_data = pd.read_csv(r'C:/Users/Lenovo/Desktop/Мартин/MIG/III година/Предмети/Интелигентни системи/Python/Проект/Databases/DB-Harlfoxem-Train.csv')
train_data['date'] = pd.to_numeric(train_data['date'].str.replace('T000000', ''))
train_data = train_data.dropna(axis=0)

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_basement', 'yr_built']
X_train = train_data[features]
Y_train = train_data['price']

test_data = pd.read_csv(r'C:/Users/Lenovo/Desktop/Мартин/MIG/III година/Предмети/Интелигентни системи/Python/Проект/Databases/DB-Harlfoxem-Test.csv')
test_data['date'] = pd.to_numeric(test_data['date'].str.replace('T000000', ''))
X_test = test_data[features]
Y_test = test_data['price']

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
predictions = lin_reg.predict(X_test)
r_squared, mae, rmse = evaluation(Y_test, predictions)
print(predictions)
print("Linear Regression -", "R^2:", r_squared, "  ", "MAE:", mae, "  ", "RMSE:", rmse)

# Ridge Regression
ridge = Ridge()
ridge.fit(X_train, Y_train)
predictions = ridge.predict(X_test)
r_squared, mae, rmse = evaluation(Y_test, predictions)
print("Ridge Regression -", "R^2:", r_squared, "  ", "MAE:", mae, "  ", "RMSE:", rmse)

# Lasso Regression
lasso = Lasso()
lasso.fit(X_train, Y_train)
predictions = lasso.predict(X_test)
r_squared, mae, rmse = evaluation(Y_test, predictions)
print("Lasso Regression -", "R^2:", r_squared, "  ", "MAE:", mae, "  ", "RMSE:", rmse)

# Elastic Net
elasticNet = ElasticNet()
elasticNet.fit(X_train, Y_train)
predictions = elasticNet.predict(X_test)
r_squared, mae, rmse = evaluation(Y_test, predictions)
print("Elastic Net -", "R^2:", r_squared, "  ", "MAE:", mae, "  ", "RMSE:", rmse)

# Least Angle Regression
lar_reg = Lars()
lar_reg.fit(X_train, Y_train)
predictions = lar_reg.predict(X_test)
r_squared, mae, rmse = evaluation(Y_test, predictions)
print("Least Angle Regression -", "R^2:", r_squared, "  ", "MAE:", mae, "  ", "RMSE:", rmse)

# Lasso Least Angle Regression
las_lar = LassoLars()
las_lar.fit(X_train, Y_train)
predictions = las_lar.predict(X_test)
r_squared, mae, rmse = evaluation(Y_test, predictions)
print("Lasso Least Angle Regression -", "R^2:", r_squared, "  ", "MAE:", mae, "  ", "RMSE:", rmse)

# Orthogonal Matching Pursuit
omp = OrthogonalMatchingPursuit()
omp.fit(X_train, Y_train)
predictions = omp.predict(X_test)
r_squared, mae, rmse = evaluation(Y_test, predictions)
print("Orthogonal Matching Pursuit -", "R^2:", r_squared, "  ", "MAE:", mae, "  ", "RMSE:", rmse)

# Bayesian Regression
bay = BayesianRidge()
bay.fit(X_train, Y_train)
predictions = bay.predict(X_test)
r_squared, mae, rmse = evaluation(Y_test, predictions)
print("Bayesian Ridge -", "R^2:", r_squared, "  ", "MAE:", mae, "  ", "RMSE:", rmse)

# KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(X_train, Y_train)
predictions = knr.predict(X_test)
r_squared, mae, rmse = evaluation(Y_test, predictions)
print("KNeighborsRegressor -", "R^2:", r_squared, "  ", "MAE:", mae, "  ", "RMSE:", rmse)

# PLS Regression
pls = PLSRegression()
pls.fit(X_train, Y_train)
predictions = pls.predict(X_test)
r_squared, mae, rmse = evaluation(Y_test, predictions)
print("PLS -", "R^2:", r_squared, "  ", "MAE:", mae, "  ", "RMSE:", rmse)

# Decision Trees
dtr = DecisionTreeRegressor()
dtr.fit(X_train, Y_train)
predictions = dtr.predict(X_test)
r_squared, mae, rmse = evaluation(Y_test, predictions)
print("DecisionTreeRegressor -", "R^2:", r_squared, "  ", "MAE:", mae, "  ", "RMSE:", rmse)

# XGB Regression
xgb = XGBRegressor()
xgb.fit(X_train, Y_train)
predictions = xgb.predict(X_test)
r_squared, mae, rmse = evaluation(Y_test, predictions)
print("XGB -", "R^2:", r_squared, "  ", "MAE:", mae, "  ", "RMSE:", rmse)

# Random Forest Regression
rfr = RandomForestRegressor()
rfr.fit(X_train, Y_train)
predictions = rfr.predict(X_test)
r_squared, mae, rmse = evaluation(Y_test, predictions)
print("Random Forest -", "R^2:", r_squared, "  ", "MAE:", mae, "  ", "RMSE:", rmse)
