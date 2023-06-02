import matplotlib.pyplot as plt
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
r_squared1, mae1, rmse1 = evaluation(Y_test, predictions)

# Ridge Regression
ridge = Ridge()
ridge.fit(X_train, Y_train)
predictions = ridge.predict(X_test)
r_squared2, mae2, rmse2 = evaluation(Y_test, predictions)

# Lasso Regression
lasso = Lasso()
lasso.fit(X_train, Y_train)
predictions = lasso.predict(X_test)
r_squared3, mae3, rmse3 = evaluation(Y_test, predictions)

# Elastic Net
elasticNet = ElasticNet()
elasticNet.fit(X_train, Y_train)
predictions = elasticNet.predict(X_test)
r_squared4, mae4, rmse4 = evaluation(Y_test, predictions)

# Least Angle Regression
lar_reg = Lars()
lar_reg.fit(X_train, Y_train)
predictions = lar_reg.predict(X_test)
r_squared5, mae5, rmse5 = evaluation(Y_test, predictions)

# Lasso Least Angle Regression
las_lar = LassoLars()
las_lar.fit(X_train, Y_train)
predictions = las_lar.predict(X_test)
r_squared6, mae6, rmse6 = evaluation(Y_test, predictions)

# Orthogonal Matching Pursuit
omp = OrthogonalMatchingPursuit()
omp.fit(X_train, Y_train)
predictions = omp.predict(X_test)
r_squared7, mae7, rmse7 = evaluation(Y_test, predictions)

# Bayesian Regression
bay = BayesianRidge()
bay.fit(X_train, Y_train)
predictions = bay.predict(X_test)
r_squared8, mae8, rmse8 = evaluation(Y_test, predictions)

# KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(X_train, Y_train)
predictions = knr.predict(X_test)
r_squared9, mae9, rmse9 = evaluation(Y_test, predictions)

# PLS Regression
pls = PLSRegression()
pls.fit(X_train, Y_train)
predictions = pls.predict(X_test)
r_squared10, mae10, rmse10 = evaluation(Y_test, predictions)

# Decision Trees
dtr = DecisionTreeRegressor()
dtr.fit(X_train, Y_train)
predictions = dtr.predict(X_test)
r_squared11, mae11, rmse11 = evaluation(Y_test, predictions)

# XGB Regression
xgb = XGBRegressor()
xgb.fit(X_train, Y_train)
predictions = xgb.predict(X_test)
r_squared12, mae12, rmse12 = evaluation(Y_test, predictions)

# Random Forest Regression
rfr = RandomForestRegressor()
rfr.fit(X_train, Y_train)
predictions = rfr.predict(X_test)
r_squared13, mae13, rmse13 = evaluation(Y_test, predictions)

# Data for the bar graph
categories = ['Lin.Reg.', 'RidgeReg.', 'LassoReg.', 'ElasticNet', 'LeastAng.Reg.', 'LassoLAR.', 'Orth.Match.', 'Bay.Reg.', 'KNeigh.', 'PLSReg.', 'Dec.Trees', 'XGBReg.', 'Rand.ForestReg.']
subcategories = ['R^2', 'MAE', 'RMSE']
values = [[r_squared1*100000, mae1, rmse1], [r_squared2*100000, mae2, rmse2], [r_squared3*100000, mae3, rmse3], [r_squared4*100000, mae4, rmse4], [r_squared5*100000, mae5, rmse5], [r_squared6*100000, mae6, rmse6], [r_squared7*100000, mae7, rmse7], [r_squared8*100000, mae8, rmse8], [r_squared9*100000, mae9, rmse9], [r_squared10*100000, mae10, rmse10], [r_squared11*100000, mae11, rmse11], [r_squared12*100000, mae12, rmse12], [r_squared13*100000, mae13, rmse13]]

# Set the width of each bar and subpart
bar_width = 0.25

# Calculate the positions of the bars on the x-axis
bar_positions = np.arange(len(categories))

# Create the figure and axis
fig, ax = plt.subplots()

# Plot the bars for each subpart
for i in range(len(subcategories)):
    subpart_values = [v[i] for v in values]
    subpart_positions = bar_positions + (i * bar_width)
    ax.bar(subpart_positions, subpart_values, bar_width, label=subcategories[i])

# Set the x-axis tick positions and labels
ax.set_xticks(bar_positions + bar_width)
ax.set_xticklabels(categories)

# Add labels and title
ax.set_xlabel('Algorithms')
ax.set_ylabel('Values')
ax.set_title('Comparison of regression algortihms')

# Add a legend
ax.legend()

# Display the graph
plt.show()
