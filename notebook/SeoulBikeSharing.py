
# ==========================================
# REGRESSION MODELLING
# SEOUL BIKE SHARING DEMAND
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures,
    OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.dummy import DummyRegressor

df = pd.read_csv("SeoulBikeData.csv", encoding="unicode_escape")
# 1. Prepare data
model_df = df[df['Functioning Day'] == 'Yes'].copy()

model_df['Date'] = pd.to_datetime(
    model_df['Date'],
    format='%d/%m/%Y'
)

model_df = model_df.sort_values(
    ['Date', 'Hour']
).reset_index(drop=True)

# 2. Cyclical feature engineering
model_df['Hour_sin'] = np.sin(2 * np.pi * model_df['Hour'] / 24)
model_df['Hour_cos'] = np.cos(2 * np.pi * model_df['Hour'] / 24)

model_df['Month'] = model_df['Date'].dt.month
model_df['Month_sin'] = np.sin(2 * np.pi * (model_df['Month'] - 1) / 12)
model_df['Month_cos'] = np.cos(2 * np.pi * (model_df['Month'] - 1) / 12)

model_df['DayOfWeek'] = model_df['Date'].dt.dayofweek
model_df['Day_sin'] = np.sin(2 * np.pi * model_df['DayOfWeek'] / 7)
model_df['Day_cos'] = np.cos(2 * np.pi * model_df['DayOfWeek'] / 7)

# 3. Define target and features
target = 'Rented Bike Count'

numeric_features = [
    'Temperature(°C)',
    'Humidity(%)',
    'Wind speed (m/s)',
    'Visibility (10m)',
    'Solar Radiation (MJ/m2)',
    'Rainfall(mm)',
    'Snowfall (cm)',
    'Hour_sin',
    'Hour_cos',
    'Month_sin',
    'Month_cos',
    'Day_sin',
    'Day_cos'
]

categorical_features = ['Holiday']

X = model_df[numeric_features + categorical_features]
y = model_df[target]

# 4. Chronological 80:20 split
split_index = int(len(model_df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training period:",
      model_df['Date'].iloc[0],
      "to",
      model_df['Date'].iloc[split_index - 1])
print("Testing period:",
      model_df['Date'].iloc[split_index],
      "to",
      model_df['Date'].iloc[-1])

# 5. Evaluation function
results = []

def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2
    })

    print("\n", name)
    print("MAE:", round(mae, 3))
    print("MSE:", round(mse, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 4))


# ==========================================
# MODEL 1: SIMPLE LINEAR REGRESSION
# ==========================================

X_simple = model_df[['Temperature(°C)']]

X_simple_train = X_simple.iloc[:split_index]
X_simple_test = X_simple.iloc[split_index:]

simple_model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

simple_model.fit(X_simple_train, y_train)

evaluate_model(
    "Simple Linear Regression",
    simple_model,
    X_simple_test,
    y_test
)


# ==========================================
# MODEL 2: MULTIPLE LINEAR REGRESSION
# ==========================================

numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transformer = OneHotEncoder(
    handle_unknown='ignore'
)

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

multiple_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

multiple_model.fit(X_train, y_train)

evaluate_model(
    "Multiple Linear Regression",
    multiple_model,
    X_test,
    y_test
)


# ==========================================
# MODEL 3: POLYNOMIAL REGRESSION
# ==========================================

polynomial_transformer = Pipeline([
    ('scaler', StandardScaler()),
    ('polynomial', PolynomialFeatures(
        degree=2,
        include_bias=False
    ))
])

poly_preprocessor = ColumnTransformer([
    ('num', polynomial_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

polynomial_model = Pipeline([
    ('preprocessor', poly_preprocessor),
    ('regressor', LinearRegression())
])

polynomial_model.fit(X_train, y_train)

evaluate_model(
    "Polynomial Regression",
    polynomial_model,
    X_test,
    y_test
)


# ==========================================
# BASELINE MODEL
# ==========================================

baseline_model = DummyRegressor(strategy='mean')
baseline_model.fit(X_train, y_train)

evaluate_model(
    "Baseline Mean Model",
    baseline_model,
    X_test,
    y_test
)


# ==========================================
# FINAL MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

print("\n\nFINAL MODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "regression_model_results.csv",
    index=False
)

# ==========================================
# REGRESSION MODEL VALIDATION
# ==========================================



# Polynomial predictions
poly_predictions = polynomial_model.predict(X_test)

# Simple predictions
simple_predictions = simple_model.predict(X_simple_test)

# Multiple predictions
multiple_predictions = multiple_model.predict(X_test)

# 1. Actual vs Predicted
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    poly_predictions,
    alpha=0.35
)

min_val = min(y_test.min(), poly_predictions.min())
max_val = max(y_test.max(), poly_predictions.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    'r--'
)

plt.xlabel("Actual Bike Rentals")
plt.ylabel("Predicted Bike Rentals")
plt.title("Actual vs Predicted - Polynomial Regression")
plt.show()


# 2. Residual Distribution
residuals = y_test - poly_predictions

plt.figure(figsize=(9, 5))

sns.histplot(
    residuals,
    bins=40,
    kde=True
)

plt.axvline(0, color='red', linestyle='--')

plt.title("Residual Distribution - Polynomial Regression")
plt.xlabel("Prediction Error")
plt.show()


# 3. Residuals vs Predicted
plt.figure(figsize=(9, 5))

plt.scatter(
    poly_predictions,
    residuals,
    alpha=0.35
)

plt.axhline(0, color='red', linestyle='--')

plt.xlabel("Predicted Bike Rentals")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.show()


# 4. Compare actual and predicted demand
comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': poly_predictions
})

comparison = comparison.reset_index(drop=True)

plt.figure(figsize=(13, 5))

plt.plot(
    comparison['Actual'].iloc[:200],
    label='Actual',
    alpha=0.8
)

plt.plot(
    comparison['Predicted'].iloc[:200],
    label='Predicted',
    alpha=0.8
)

plt.title("Actual vs Predicted Bike Demand (First 200 Test Records)")
plt.xlabel("Test Observation")
plt.ylabel("Bike Rentals")
plt.legend()
plt.show()


# 5. Compare training and testing target distributions
print("\nTARGET DISTRIBUTION COMPARISON")

print("Training mean:", round(y_train.mean(), 2))
print("Testing mean:", round(y_test.mean(), 2))

print("Training median:", round(y_train.median(), 2))
print("Testing median:", round(y_test.median(), 2))

print("Training standard deviation:", round(y_train.std(), 2))
print("Testing standard deviation:", round(y_test.std(), 2))

print("\nRESIDUAL SUMMARY")
print(pd.Series(residuals).describe())
