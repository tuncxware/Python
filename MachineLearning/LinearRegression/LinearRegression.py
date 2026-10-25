import matplotlib.pyplot as plt
print("Hello, welcome to the Linear Regression module!")

x = [1, 2, 3, 4, 5] # Study Hours
y = [50, 55, 65, 70 ,80] # Actual Scores

# Formula for Linear Regression: y = mx + b

m = (len(x) * sum([x[i] * y[i] for i in range(len(x))]) - sum(x) * sum(y)) / (len(x) * sum([i**2 for i in x]) - sum(x)**2)
b = (sum(y) - m * sum(x)) / len(x)

print(f"The calculated slope (m) is: {m}")
print(f"The calculated y-intercept (b) is: {b}")
print(f"The linear regression equation is: y_pred = {m}x + {b}")

predicted_scores = [m * i + b for i in x]
print(f"The predicted scores based on the linear regression model are: {predicted_scores}")

MAE = sum([abs(y[i] - predicted_scores[i]) for i in range(len(y))]) / len(y)
print(f"The Mean Absolute Error (MAE) of the model is: {MAE}")

RMSE = (sum([(y[i] - predicted_scores[i])**2 for i in range(len(y))]) / len(y))**0.5
print(f"The Root Mean Squared Error (RMSE) of the model is: {RMSE}")

R_squared = 1 - (sum([(y[i] - predicted_scores[i])**2 for i in range(len(y))]) / sum([(y[i] - sum(y)/len(y))**2 for i in range(len(y))]))
print(f"The R-squared value of the model is: {R_squared}")

# Plotting the data points and the regression line
plt.scatter(x, y, color='blue', label='Actual Scores')
plt.plot(x, predicted_scores, color='red', label='Regression Line')
plt.vlines(x, predicted_scores, y, colors='gray', linestyles='dashed', label='Errors (Residuals)')
plt.xlabel('Study Hours')
plt.ylabel('Actual Scores')
plt.title('Linear Regression')
plt.legend()
plt.show()