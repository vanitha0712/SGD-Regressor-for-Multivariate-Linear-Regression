# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required packages and print the present data.
2.Print the placement data and salary data. 
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices. 5.Display the results.
```

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: B.vanitha
RegisterNumber: 212225220117
*/

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
X = np.array([
    [1000, 2],
    [1500, 3],
    [1800, 4],
    [2400, 3],
    [3000, 5]
])

y = np.array([
    [200000, 3],
    [300000, 4],
    [350000, 5],
    [450000, 4],
    [600000, 6]
])

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

models = []
for i in range(y_scaled.shape[1]):
    model = SGDRegressor(max_iter=2000, eta0=0.01)
    model.fit(X_scaled, y_scaled[:, i])
    models.append(model)

y_pred_scaled = np.column_stack([m.predict(X_scaled) for m in models])
y_pred = scaler_y.inverse_transform(y_pred_scaled)


plt.figure()

plt.scatter(y[:,0], y_pred[:,0])

min_val = min(y[:,0].min(), y_pred[:,0].min())
max_val = max(y[:,0].max(), y_pred[:,0].max())
plt.plot([min_val, max_val], [min_val, max_val])  # y = x line

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (SGD Regressor)")

plt.show()
new_data = np.array([[2000, 3]])
new_scaled = scaler_X.transform(new_data)

pred_scaled = [m.predict(new_scaled)[0] for m in models]
pred = scaler_y.inverse_transform([pred_scaled])

print("Predicted Price:", pred[0][0])
print("Predicted Occupants:", pred[0][1])


```

## Output:
<img width="823" height="617" alt="image" src="https://github.com/user-attachments/assets/caa98542-bc83-47b3-8f58-56bdaad7bacf" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
