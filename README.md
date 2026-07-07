 Global Temperature Analysis & Prediction

 Project Overview

This project analyzes historical global temperature data and predicts **Land and Ocean Average Temperature** using Machine Learning. The project includes data cleaning, preprocessing, exploratory data analysis (EDA), feature engineering, model building, and model evaluation.

---

 Objectives

* Analyze historical global temperature trends.
* Clean and preprocess climate data.
* Perform exploratory data analysis (EDA).
* Build a machine learning model for temperature prediction.
* Evaluate model performance using regression metrics.

---

 Dataset

* **Dataset Name:** GlobalTemperatures.csv
* **Type:** Historical Climate Data

---

 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

 Project Workflow

 1. Data Loading

* Imported the dataset using Pandas.

 2. Data Cleaning

* Removed unnecessary columns.
* Converted temperatures from Celsius to Fahrenheit.
* Converted date columns into datetime format.
* Extracted the Year feature.
* Removed missing values.

 3. Exploratory Data Analysis (EDA)

* Generated descriptive statistics.
* Created a correlation heatmap.
* Analyzed relationships between temperature variables.

 4. Feature Engineering

Features:

* Land Average Temperature
* Land Maximum Temperature
* Land Minimum Temperature

Target:

* Land and Ocean Average Temperature

 5. Model Building

* SelectKBest
* StandardScaler
* RandomForestRegressor

 6. Model Evaluation

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Prediction Accuracy

---

 Results

The Random Forest Regression model successfully predicted global temperatures with good accuracy after data preprocessing and feature engineering.

---

## 📁 Project Structure

```text
global-temperature-analysis-prediction/
│── data/
│── notebook.ipynb (or main.py)
│── images/
│── README.md
│── requirements.txt
```

---

 Future Improvements

* Hyperparameter tuning
* Time-series forecasting models
* Interactive dashboard
* Model deployment

---

 Project Visualizations

### Correlation Heatmap

![Correlation Heatmap](images/correlation_heatmap.png)
