import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Position_Salaries.csv")

X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=0)
tree_reg.fit(X, y)
X_grid = np.arange(min(X), max(X), 0.01).reshape(-1, 1)

y_pred = tree_reg.predict(X_grid)

plt.scatter(X, y, color='red', label='Actual Salary')

# Plot the predicted values from the Decision Tree
plt.plot(X_grid, y_pred, color='blue', label='Predicted Salary')

plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()

# Show the plot
plt.show()
