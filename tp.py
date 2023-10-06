import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2

#Numpy

np.random.seed(42)

X = 3 * np.random.rand(1000)

moyenne_X = round(np.mean(X), 2)
ecart_type_X = round(np.std(X), 2)
mediane_X = round(np.median(X), 2)

print("\nX:")
print(f"Moyenne de X: {moyenne_X}")
print(f"Écart type de X: {ecart_type_X}")
print(f"Médiane de X: {mediane_X}\n")

X_bis = 3 * np.random.rand(1000)

moyenne_X_bis = round(np.mean(X_bis), 2)
ecart_type_X_bis = round(np.std(X_bis), 2)
mediane_X_bis = round(np.median(X_bis), 2)

print("X_bis:")
print(f"Moyenne de X_bis: {moyenne_X_bis}")
print(f"Écart type de X_bis: {ecart_type_X_bis}")
print(f"Médiane de X_bis: {mediane_X_bis}\n")

noise = 0.1 * np.random.randn(1000)
y = np.sin(X) + noise


#Matplotlib

plt.figure(figsize=(8,6))
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('y en fonction de X')
plt.show()

plt.figure(figsize=(8,6))
plt.hist(noise, bins=50)
plt.xlabel('Valeur du bruit')
plt.ylabel('Fréquence')
plt.title('Distribution du bruit gaussien')
plt.show()
