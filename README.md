# Linear Regression Implementation in Python

## Introduction

Linear Regression is a supervised learning algorithm widely used for predicting a continuous outcome variable (dependent variable) based on one or more predictor variables (independent variables). In simple linear regression, there is one predictor variable, while in multiple linear regression, there are two or more.

The goal of linear regression is to find the linear relationship between the input features and the target variable. The equation of a simple linear regression model is given by:

$$( Y = wX + b )$$

where:
- $( Y )$ is the target variable,
- $( X )$ is the predictor variable,
- $( w )$ is the slope (coefficient) of the line,
- $( b )$ is the y-intercept.

For multiple linear regression with $n$ features:

$[ Y = b_0 + w_1X_1 + w_2X_2 + \ldots + w_nX_n ]$

where:
- $( Y )$ is the target variable,
- $( b_0 )$ is the y-intercept,
- $( w_1, w_2, \ldots, w_n )$ are the coefficients,
- $( X_1, X_2, \ldots, X_n )$ are the predictor variables.

## Cost Function: Mean Squared Error

The mean squared error (MSE) is a measure of the average squared difference between the actual and predicted values. It is a widely used cost function for linear regression. The formula for MSE is given by:

$$[ MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 ]$$

where:
- $( m )$ is the number of samples,
- $( y_i )$ is the actual target value for the \( i^{th} \) sample,
- $( \hat{y}_i )$ is the predicted target value for the \( i^{th} \) sample.

## Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the cost function (MSE in the case of linear regression). The gradient represents the direction of the steepest increase of the cost function, and moving in the opposite direction of the gradient reduces the cost. The update rules for the weights \( w \) and \( b \) in the context of gradient descent are given by:

$[ w := w - \alpha \frac{\partial}{\partial w}MSE ]$

$[ b := b - \alpha \frac{\partial}{\partial b}MSE ]$

where:
- $( \alpha )$ is the learning rate.

## Implementation

This project implements linear regression from scratch in Python, including the calculation of the mean squared error, the use of gradient descent for optimization, and the visualization of the regression line.

#### Training the Model:

The `linear_regression` function is used to train the model on the provided data. It iteratively updates weights and bias using gradient descent until convergence or a specified number of iterations.

#### Making Predictions:

Once the model is trained, predictions are made using the learned parameters. In the example, the predicted values (\(Y_{\text{pred}}\)) are computed using the formula \(Y_{\text{pred}} = Xw + b\).

#### Evaluation:

The model's performance is evaluated using the Mean Squared Error (MSE), providing a quantitative measure of the prediction accuracy.

#### Visualization:

Matplotlib and Plotly are used for visualizing the data points and the regression line in both 2D and 3D plots.


## Usage

To use the implementation, follow these steps:

1. Import the necessary libraries (NumPy, Matplotlib, etc.).
2. Load or generate your dataset.
3. Use the `linear_regression` function to train the model on your data.
4. Make predictions using the learned parameters.
5. Evaluate the model using metrics such as mean squared error.

Refer to the code and comments for detailed usage instructions.

## Example

Here's a brief example of how to use the linear regression implementation:

```python
import numpy as np
from linear_regression import linear_regression, mean_squared_error

# Load or generate your dataset (X, Y)

# Train the model
estimated_weights, estimated_bias = linear_regression(X, Y)

# Make predictions
predictions = np.dot(X, estimated_weights) + estimated_bias

# Evaluate the model
mse = mean_squared_error(Y, predictions)
print(f"Mean Squared Error: {mse}")
```
