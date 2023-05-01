"""Step 2 Linear regression with many independent variables"""

import pandas as pd
from sklearn.linear_model import LinearRegression,  LassoCV
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from scipy import stats
import numpy as np


# Step 1: Download the data using pandas.read_csv
data = pd.read_csv("data.csv")

# Step 2: Make x a data frame with predictors and y a series with salary
# Remove the target variable from the data
x = data.drop("salary", axis=1)
# One-hot encode categorical variables
x = pd.get_dummies(x)
# Select the target variable
y = data["salary"]

# Step 3: Divide the predictors and the target into training and test parts
# Use test_size=0.3 and random_state=100
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# Step 4: Feature selection using LassoCV
scaler = StandardScaler()
lasso = LassoCV(cv=5, random_state=100)
sfm = SelectFromModel(lasso)

pipe = make_pipeline(scaler, sfm, LinearRegression())

pipe.fit(x_train, y_train)

# Step 5: Predict wages using the fitted model based on the test data and calculate the MAPE
y_pred = pipe.predict(x_test)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Step 6: Print three floating-point numbers: b0 and b1 coefficients and MAPE,
# rounded to five decimal places and separated by spaces
b0 = pipe.named_steps['linearregression'].intercept_
b1 = pipe.named_steps['linearregression'].coef_[0]
print(f"{b0:.5f} {b1:.5f} {mape:.5f}")

# Draw a scatter plot showing the relationship between predictors and salary
x_test_numeric = x_test.select_dtypes(include=[np.number])
if not x_test_numeric.empty:
    for col in x_test_numeric.columns:
        # Get the slope, intercept, and other statistics of the linear regression
        slope, intercept, r, p, std_err = stats.linregress(x_test[col], y_test)

        # Define a function that represents the regression line
        def myfunc(x_func):
            return slope * x_func + intercept
        # Apply the function to the x_func values and get the y values for the line
        mymodel = list(map(myfunc, x_test[col]))
        # Plot the scatter plot and the regression line
        plt.scatter(x_test[col], y_test, color="blue", label="Actual response")
        plt.plot(x_test[col], mymodel, color="red", label="Estimated regression line f(x)")
        plt.xlabel(col)
        plt.ylabel("salary")
        plt.legend()
        plt.show()
else:
    print("No numeric columns found in x_test dataframe.")

# Add numeric analysis of data and model using print statements
# Print the number of observations in the dataset
print("Number of observations:", len(data))

# Print descriptive statistics for the variables
print("Descriptive statistics for variables:")
print(data.describe())

# Select only the numerical columns for the correlation matrix calculation
num_cols = data.select_dtypes(include=["int64", "float64"]).columns
corr_matrix = data[num_cols].corr()

# Print the correlation matrix for the numerical variables
print("Correlation matrix for variables:")
print(corr_matrix)

# Print the R-squared for the model on the test data
print("R-squared for the model:")
print(pipe.score(x_test, y_test))

# Print the coefficients of the model
print("Coefficients of the model:")
for idx, col_name in enumerate(x_train.columns):
    coef = pipe.named_steps['linearregression'].coef_
    if idx < len(coef):
        print(f"{col_name}: {coef[idx]}")
print(f"Intercept: {pipe.named_steps['linearregression'].intercept_}")
