import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import os
from PIL import Image
from sklearn.model_selection import train_test_split

#Mise en place de l’environnement de code

##Numpy

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


##Matplotlib

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

#Données

dossier_bike = "data/bike"
nombre_fichier_bikes = len(os.listdir(dossier_bike))
print(f"Nombre de fichiers dans le dossier bike: {nombre_fichier_bikes}")

dossier_car = "data/car"
nombre_fichier_cars = len(os.listdir(dossier_car))
print(f"Nombre de fichiers dans le dossier car: {nombre_fichier_cars}")

print(f"Nombre de fichiers total: {nombre_fichier_bikes + nombre_fichier_cars}")

liste_image = []

for nom_fichier in os.listdir(dossier_bike):
    chemin_complet = os.path.join(dossier_bike, nom_fichier)
    with Image.open(chemin_complet) as img:
        largeur, hauteur = img.size
        format_image = img.format
        taille_image = img.size
        
        print(f"Fichier : {nom_fichier}")
        print(f"Format : {format_image}")
        print(f"Dimensions : {largeur}x{hauteur}")
        print("-----")
        liste_image.append((nom_fichier,chemin_complet,format_image,taille_image))
        
                
for nom_fichier in os.listdir(dossier_car):
    chemin_complet = os.path.join(dossier_car, nom_fichier)
    with Image.open(chemin_complet) as img:
        largeur, hauteur = img.size
        format_image = img.format
        taille_image = img.size
        
        print(f"Fichier : {nom_fichier}")
        print(f"Format : {format_image}")
        print(f"Dimensions : {largeur}x{hauteur}")
        print("-----")
        liste_image.append((nom_fichier,chemin_complet,format_image,taille_image))

image = plt.imread(liste_image[0][1])
plt.imshow(image[:,:,:])
plt.show()

plt.imshow(image[:,:,1], cmap="gray")
plt.show()

plt.imshow(image[:,:,1], cmap="gray", origin="lower")
plt.show()


target_size = (224,224)

def populate_images_and_labels_lists(image_folder_path, label):
    images = []
    labels = []
    
    for filename in os.listdir(image_folder_path):
       
        image = cv2.imread(os.path.join(image_folder_path, filename))
        resized_image = cv2.resize(image, target_size)

        images.append(resized_image)
        labels.append(label)
        
    return images, labels

bike_images, bike_labels = populate_images_and_labels_lists(dossier_bike, "bike")
car_images, car_labels = populate_images_and_labels_lists(dossier_car, "car")

all_images = bike_images + car_images
all_labels = bike_labels + car_labels

images_array = np.array(all_images)
labels_array = np.array(all_labels)

images = np.array([image.flatten() for image in images_array])

X_train, X_test, y_train, y_test = train_test_split(images, labels_array, test_size=0.2, random_state=0)

