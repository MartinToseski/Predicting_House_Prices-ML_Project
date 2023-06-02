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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

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

#Grid Search
"""
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 5, 10],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    # Add more hyperparameters as needed
}

model = RandomForestRegressor()
grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search.fit(X_train, Y_train)
# Get the best hyperparameters and corresponding score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
r_squared, mae, rmse = evaluation(Y_test, predictions)
print("Random Forest -", "R^2:", r_squared, "  ", "MAE:", mae, "  ", "RMSE:", rmse)
"""

#BayesianOptimization

# Define the objective function to optimize
def rf_cv(n_estimators, max_depth, min_samples_split):
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split)
    )
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    return -mean_squared_error(Y_test, predictions)

# Define the search space
param_bounds = {
    'n_estimators': (100, 300),
    'max_depth': (None, 10),
    'min_samples_split': (2, 10)
}

# Create the BayesianOptimization object and perform optimization
optimizer = BayesianOptimization(f=rf_cv, pbounds=param_bounds, verbose=2, random_state=42)
optimizer.maximize(init_points=5, n_iter=10)

# Get the best hyperparameters and corresponding score
best_params = optimizer.max['params']
best_model = RandomForestRegressor(
    n_estimators=int(best_params['n_estimators']),
    max_depth=int(best_params['max_depth']),
    min_samples_split=int(best_params['min_samples_split'])
)

best_model.fit(X_train, Y_train)
predictions = best_model.predict(X_test)
r_squared, mae, rmse = evaluation(Y_test, predictions)
print("Random Forest -", "R^2:", r_squared, "  ", "MAE:", mae, "  ", "RMSE:", rmse)
