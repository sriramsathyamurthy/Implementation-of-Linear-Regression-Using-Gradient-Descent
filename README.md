# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Data
# -----------------------
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

# -----------------------
# Parameters
# -----------------------
w = 0.0
b = 0.0
alpha = 0.01
epochs = 100
n = len(x)

losses = []

# -----------------------
# Gradient Descent
# -----------------------
for _ in range(epochs):
    y_hat = w * x + b

    # Mean Squared Error
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    w -= alpha * dw
    b -= alpha * db

# -----------------------
# Plots
# -----------------------
plt.figure(figsize=(12, 5))

# 1️⃣ Loss vs Iterations
plt.subplot(1, 2, 1)
plt.plot(losses, color="blue")
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

# 2️⃣ Regression Line
plt.subplot(1, 2, 2)
plt.scatter(x, y, color="red", label="Data")
plt.plot(x, w * x + b, color="green", label="Regression Line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()

plt.tight_layout()
plt.show()

print("Final weight (w):", w)
print("Final bias (b):", b)
/*
Program to implement the linear regression using gradient descent.
Developed by: SRIRAM S
RegisterNumber:  212225240155
*/
```

## Output:
<img width="1370" height="582" alt="image" src="https://github.com/user-attachments/assets/7a24252f-8236-454a-90da-1ef2247cd655" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
