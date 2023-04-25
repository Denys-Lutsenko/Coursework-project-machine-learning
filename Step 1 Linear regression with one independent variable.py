"""Step 1 Linear regression with one independent variable"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import pearsonr, levene

# Load data from csv file
df = pd.read_csv('data.csv')

# Create x as a dataframe with the rating column and y as a series with the salary column
x = df[['rating']]
y = df['salary']

# Split predictor x and target variable y into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# Train a linear regression model on the training data, predicting salary based on rating
model = LinearRegression()
model.fit(x_train, y_train)

# Predict salary using the trained model based on the testing data and calculate MAPE
y_pred = model.predict(x_test)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Output three floating-point numbers: intercept b0, slope b1, and MAPE,
# rounded to five decimal places and separated by a space
b0 = model.intercept_
b1 = model.coef_[0]
print(f'{b0:.5f} {b1:.5f} {mape:.5f}')

# Create a scatter plot showing the relationship between rating and salary and its linear approximation
plt.scatter(x_test, y_test, color='blue')
plt.plot(x_test, y_pred, color='black')
plt.xlabel('Rating')
plt.ylabel('Salary')
plt.show()

# Analyze the data using print statements

# Calculate the Pearson correlation coefficient between rating and salary
corr, p_value = pearsonr(x_test['rating'], y_test)
print(f'Pearson correlation coefficient: {corr:.5f}')
print(f'P-value: {p_value:.5f}')

# Test the hypothesis of equal variances of salaries in two groups based on rating (above and below the mean)
rating_mean = x_test['rating'].mean()
group1 = y_test[x_test['rating'] < rating_mean]
group2 = y_test[x_test['rating'] >= rating_mean]
stat, p_value = levene(group1, group2)
print(f'Levene statistic: {stat:.5f}')
print(f'P-value: {p_value:.5f}')
