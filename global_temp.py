import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# -----------------------------
# Load Dataset
# -----------------------------
global_temp = pd.read_csv(r"C:\Users\swati singh\OneDrive\Desktop\DATA\GlobalTemperatures.csv")

print(global_temp.head())
print(global_temp.shape)
print(global_temp.columns)
global_temp.info()
print(global_temp.isnull().sum())


# -----------------------------
# Temperature Conversion Function
# -----------------------------
def convert_temp(celsius):
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit


# -----------------------------
# Data Preparation (Wrangling)
# -----------------------------
def wrangle(df):
    df = df.copy()

    # Drop uncertainty columns
    df = df.drop(columns=[
        "LandAverageTemperatureUncertainty",
        "LandMaxTemperatureUncertainty",
        "LandMinTemperatureUncertainty",
        "LandAndOceanAverageTemperatureUncertainty"
    ])

    # Convert Celsius → Fahrenheit
    df["LandAverageTemperature"] = df["LandAverageTemperature"].apply(convert_temp)
    df["LandMaxTemperature"] = df["LandMaxTemperature"].apply(convert_temp)
    df["LandMinTemperature"] = df["LandMinTemperature"].apply(convert_temp)
    df["LandAndOceanAverageTemperature"] = df["LandAndOceanAverageTemperature"].apply(convert_temp)

    # Convert date column
    df["dt"] = pd.to_datetime(df["dt"], format="mixed", dayfirst=True)

    # Extract Year and Month
    df["Year"] = df["dt"].dt.year
    df["Month"] = df["dt"].dt.month

    # Drop unnecessary columns
    df = df.drop(columns=["dt", "Month"])

    # Filter data
    df = df[df["Year"] >= 1850]

    # Set index
    df = df.set_index("Year")

    # Drop missing values
    df = df.dropna()

    return df


# -----------------------------
# Apply Wrangling
# -----------------------------
df = wrangle(global_temp)
print(global_temp.head())

## Corerlation Visualization
corrMatrix = df.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corrMatrix, annot=True, cmap="coolwarm")
plt.title("Temperature Correlation Matrix")
plt.show()

# Show cleaned data
print(df.head())
print(df.describe())

# Feature and Target Selection
target = "LandAndOceanAverageTemperature"

X = df[
    ["LandAverageTemperature",
     "LandMaxTemperature",
     "LandMinTemperature"]
]

y = df[target]


# Train Test Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

print(X_train.shape)
print(X_val.shape)


# Baseline Model
baseline_pred = [y_train.mean()] * len(y_train)

baseline_error = mean_squared_error(y_train, baseline_pred)
print("Baseline MSE:", baseline_error)


# Random Forest Model Pipeline
model = make_pipeline(
    SelectKBest(k="all"),
    StandardScaler(),
    RandomForestRegressor(
        n_estimators=100,
        max_depth=50,
        random_state=77,
        n_jobs=-1
    )
)

# Train Model
model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_val)


# Evaluation
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)

print("Model MSE:", mse)
print("Model RMSE:", rmse)


# Accuracy style metric (MAPE)
errors = abs(y_pred - y_val)
mape = 100 * (errors / y_val)
accuracy = 100 - np.mean(mape)

print("Model Accuracy:", round(accuracy, 2), "%")
