import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data set
df = pd.read_csv("crop_yield_data.csv", encoding='unicode_escape')
df = df[:100]

# Normalize the data set
attributes = ['Rain Fall (mm)', 'Fertilizer(urea) (kg/acre)', 'Temperature (°C)', 'Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
for attr in attributes:
    mean = df[attr].mean()
    std = df[attr].std()
    df[attr] = (df[attr] - mean) / std

# Split the data into training and testing sets
X = df[attributes]
y = df['Yeild (Q/acre)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=38)

# Hyperparameter Tuning with Grid Search
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=DecisionTreeRegressor(random_state=41),
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',  # Use MSE for scoring
                           cv=5)  # 5-fold cross-validation

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

best_regressor = DecisionTreeRegressor(random_state=41, **best_params)
best_regressor.fit(X_train, y_train)

# Train a Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=41)
rf_regressor.fit(X_train, y_train)
y_pred_rf = rf_regressor.predict(X_test)

# Calculate R2, MSE, and RMSE for the Random Forest model
r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)

print('Random Forest R Squared (R2) Score:', r2_rf)
print('Random Forest Mean Square Error (MSE):', mse_rf)
print('Random Forest Root Mean Square Error (RMSE):', rmse_rf)

# Feature Importance Analysis
feature_importances = best_regressor.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': attributes, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance Analysis')
plt.show()
