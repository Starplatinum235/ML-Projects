import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# dataset
data = pd.read_csv("data.csv")  # Make sure to save the provided data as 'tips.csv'
print(data.head())

# selecting variables
data = data[["total_bill", "tip", "sex", "smoker", "day", "time", "size"]]

# converting the categorical data into numerical
data = pd.get_dummies(data, columns=["sex", "smoker", "day", "time"], drop_first=True)

# Defining x(independent) and y(dependent)
x = data.drop(["tip"], axis=1)
y = data["tip"]

# splitting the dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# applying linear regression
linear = LinearRegression()

# training the model with training data
linear.fit(x_train, y_train)

# accuracy of the model
Accuracy = linear.score(x_test, y_test)
print("Accuracy of this model:", Accuracy * 100)

# intercept and co-efficient
print("Coeffecients of this model:", linear.coef_)
print("Intercept:", linear.intercept_)

# predicting tips using the testing dataset
predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print(f"Predicted: {predictions[i]}, Attributes: {x_test.iloc[i].tolist()}, Actual: {y_test.iloc[i]}")
