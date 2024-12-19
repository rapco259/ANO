from sklearn.linear_model import LinearRegression
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Generovanie dát (lineárny model)
np.random.seed(0)
x = np.linspace(0, 10, 20).reshape(-1, 1)
y = 2 * x + 1 + np.random.normal(0, 1, size=x.shape)  # Lineárna funkcia s hlukom

# Prispôsobenie modelu metódou najmenších štvorcov
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# Vizualizácia
plt.scatter(x, y, color='blue', label='Dáta')
plt.plot(x, y_pred, color='red', label='Least Squares fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Metóda najmenších štvorcov')
plt.grid()
plt.show()
