import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit 
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def evaluation(y, predictions):
    mae = mean_absolute_error(Y_test, predictions)
    mse = mean_squared_error(Y_test, predictions)
    rmse = np.sqrt(mean_squared_error(Y_test, predictions))
    r_squared = r2_score(Y_test, predictions)
    return mae, mse, rmse, r_squared

train_data = pd.read_csv(r'C:/Users/Lenovo/Desktop/Мартин/MIG/III година/Предмети/Интелигентни системи/Python/Проект/Databases/DB-Harlfoxem.csv')
train_data['date'] = pd.to_numeric(train_data['date'].str.replace('T000000', ''))
train_data = train_data.dropna(axis=0)

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_basement', 'yr_built']
X_train = train_data[features]
Y_train = train_data['price']

test_data = pd.read_csv(r'C:/Users/Lenovo/Desktop/Мартин/MIG/III година/Предмети/Интелигентни системи/Python/Проект/Databases/DB-Harlfoxem.csv')
test_data['date'] = pd.to_numeric(test_data['date'].str.replace('T000000', ''))
X_test = test_data[features]
Y_test = test_data['price']

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
predictions = lin_reg.predict(X_test)
mae, mse, rmse, r_squared = evaluation(Y_test, predictions)
print("Linear Regression -", "MAE:", mae, "MSE:", mse, "RMSE:", rmse, "R_SQUARED:", r_squared)

# Ridge Regression
ridge = Ridge()
ridge.fit(X_train, Y_train)
predictions = ridge.predict(X_test)
mae, mse, rmse, r_squared = evaluation(Y_test, predictions)
print("Ridge Regression -", "MAE:", mae, "MSE:", mse, "RMSE:", rmse, "R_SQUARED:", r_squared)

# Lasso Regression
lasso = Lasso()
lasso.fit(X_train, Y_train)
predictions = lasso.predict(X_test)
mae, mse, rmse, r_squared = evaluation(Y_test, predictions)
print("Lasso Regression -", "MAE:", mae, "MSE:", mse, "RMSE:", rmse, "R_SQUARED:", r_squared)

# Elastic Net
elasticNet = ElasticNet()
elasticNet.fit(X_train, Y_train)
predictions = elasticNet.predict(X_test)
mae, mse, rmse, r_squared = evaluation(Y_test, predictions)
print("Elastic Net -", "MAE:", mae, "MSE:", mse, "RMSE:", rmse, "R_SQUARED:", r_squared)

# Least Angle Regression
lar_reg = Lars()
lar_reg.fit(X_train, Y_train)
predictions = lar_reg.predict(X_test)
mae, mse, rmse, r_squared = evaluation(Y_test, predictions)
print("Least Angle Regression -", "MAE:", mae, "MSE:", mse, "RMSE:", rmse, "R_SQUARED:", r_squared)

# Lasso Least Angle Regression
las_lar = LassoLars()
las_lar.fit(X_train, Y_train)
predictions = las_lar.predict(X_test)
mae, mse, rmse, r_squared = evaluation(Y_test, predictions)
print("Lasso Least Angle Regression -", "MAE:", mae, "MSE:", mse, "RMSE:", rmse, "R_SQUARED:", r_squared)

# Orthogonal Matching Pursuit
omp = OrthogonalMatchingPursuit()
omp.fit(X_train, Y_train)
predictions = omp.predict(X_test)
mae, mse, rmse, r_squared = evaluation(Y_test, predictions)
print("Orthogonal Matching Pursuit -", "MAE:", mae, "MSE:", mse, "RMSE:", rmse, "R_SQUARED:", r_squared)

# Bayesian Regression
bay = BayesianRidge()
bay.fit(X_train, Y_train)
predictions = bay.predict(X_test)
mae, mse, rmse, r_squared = evaluation(Y_test, predictions)
print("Bayesian Ridge -", "MAE:", mae, "MSE:", mse, "RMSE:", rmse, "R_SQUARED:", r_squared)

# PassiveAggressiveRegressor
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
per = PassiveAggressiveRegressor()
per.fit(X_scaled, Y_train)
predictions = per.predict(X_scaled)
mae, mse, rmse, r_squared = evaluation(Y_test, predictions)
print("PassiveAggressiveRegressor -", "MAE:", mae, "MSE:", mse, "RMSE:", rmse, "R_SQUARED:", r_squared)

# RobustnessRegression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
per = HuberRegressor()
per.fit(X_scaled, Y_train)
predictions = per.predict(X_scaled)
mae, mse, rmse, r_squared = evaluation(Y_test, predictions)
print("RobustnessRegression -", "MAE:", mae, "MSE:", mse, "RMSE:", rmse, "R_SQUARED:", r_squared)

# Quarantine Regression 
X_scaled = sm.add_constant(X_train)
model = sm.QuantReg(Y_train, X_scaled)
result = model.fit(q=0.25) #0.25  0.5 0.75
coefficients = result.params
predictions = result.predict(X_scaled)
mae, mse, rmse, r_squared = evaluation(Y_test, predictions)
print("QuarantineRegression -", "MAE:", mae, "MSE:", mse, "RMSE:", rmse, "R_SQUARED:", r_squared)

# KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=10)
knr.fit(X_train, Y_train)
predictions = knr.predict(X_test)
mae, mse, rmse, r_squared = evaluation(Y_test, predictions)
print("KNeighborsRegressor -", "MAE:", mae, "MSE:", mse, "RMSE:", rmse, "R_SQUARED:", r_squared)

# PLS Regression
pls = PLSRegression(n_components=1)
pls.fit(X_train, Y_train)
predictions = pls.predict(X_test)
mae, mse, rmse, r_squared = evaluation(Y_test, predictions)
print("PLS -", "MAE:", mae, "MSE:", mse, "RMSE:", rmse, "R_SQUARED:", r_squared)

# Decision Trees
dtr = DecisionTreeRegressor()
dtr.fit(X_train, Y_train)
predictions = dtr.predict(X_test)
mae, mse, rmse, r_squared = evaluation(Y_test, predictions)
print("DecisionTreeRegressor -", "MAE:", mae, "MSE:", mse, "RMSE:", rmse, "R_SQUARED:", r_squared)

# XGB Regression
xgb = XGBRegressor()
xgb.fit(X_train, Y_train)
predictions = xgb.predict(X_test)
mae, mse, rmse, r_squared = evaluation(Y_test, predictions)
print("XGB -", "MAE:", mae, "MSE:", mse, "RMSE:", rmse, "R_SQUARED:", r_squared)
 
# Random Forest Regression
rfr = RandomForestRegressor()
rfr.fit(X_train, Y_train)
predictions = rfr.predict(X_test)
mae, mse, rmse, r_squared = evaluation(Y_test, predictions)
print("Random Forest -", "MAE:", mae, "MSE:", mse, "RMSE:", rmse, "R_SQUARED:", r_squared)

# 