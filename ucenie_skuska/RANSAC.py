import numpy as np
import cv2
import matplotlib.pyplot as plt

# Generovanie dát
np.random.seed(0)
x = np.linspace(0, 100, 100)
y = 0.5 * x + 10 + np.random.normal(0, 5, 100)  # Priamka s hlukom
outliers_x = np.random.uniform(0, 100, 10)  # Odľahlé hodnoty
outliers_y = np.random.uniform(0, 100, 10)
x = np.concatenate((x, outliers_x))
y = np.concatenate((y, outliers_y))

# RANSAC na fitovanie priamky
points = np.column_stack((x, y)).astype(np.float32)
ransac = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
vx, vy, x0, y0 = ransac.flatten()
line_x = np.linspace(0, 100, 100)
line_y = vy/vx * (line_x - x0) + y0

# Vizualizácia
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(line_x, line_y, color='red', label='RANSAC Line')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('RANSAC Line Fitting')
plt.grid()
plt.show()
