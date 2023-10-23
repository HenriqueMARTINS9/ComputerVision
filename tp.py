import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV



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

##Modèles de classification
clf = DecisionTreeClassifier(random_state=0)

X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

clf.fit(X_train_reshaped, y_train)

predicted_label = clf.predict(X_test_reshaped[0].reshape(1, -1))
print(f"Prédiction du label pour la première image du set de test : {predicted_label[0]}")


clf_svm = SVC(random_state=0)

X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

clf_svm.fit(X_train_reshaped, y_train)

predicted_label_svm = clf_svm.predict(X_test_reshaped[0].reshape(1, -1))

print(f"Prédiction du label par SVM pour la première image du set de test : {predicted_label_svm[0]}")

y_pred_tree = clf.predict(X_test_reshaped)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Accuracy du modèle Arbre de décision : {accuracy_tree:.2f}")

y_pred_svm = clf_svm.predict(X_test_reshaped)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy du modèle SVM : {accuracy_svm:.2f}")

conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)
print(conf_matrix_tree)

print(f"Nombre de 'bike' classifiés comme 'car' : {conf_matrix_tree[1][0]}")
print(f"Nombre de 'car' classifiés comme 'bike' : {conf_matrix_tree[0][1]}")

conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
print(conf_matrix_svm)

precision_tree = precision_score(y_test, y_pred_tree, pos_label='bike')
recall_tree = recall_score(y_test, y_pred_tree, pos_label='bike')

print(f"Précision du modèle Arbre de décision : {precision_tree:.2f}")
print(f"Spécificité (Recall) du modèle Arbre de décision : {recall_tree:.2f}")

y_probs_tree = clf.predict_proba(X_test_reshaped)[:, 1] 

fpr, tpr, thresholds = roc_curve(y_test, y_probs_tree, pos_label='bike')
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

## Comparaison de pipeline et fine tuning

depth = clf.get_depth()
print(f"Profondeur de l'arbre de décision : {depth}")

max_depth_list = list(range(1, 13))

train_accuracy = []
test_accuracy = []

for depth in max_depth_list:
    clf_temp = DecisionTreeClassifier(max_depth=depth, random_state=0)
    clf_temp.fit(X_train_reshaped, y_train)
    
    train_accuracy.append(accuracy_score(y_train, clf_temp.predict(X_train_reshaped)))
    test_accuracy.append(accuracy_score(y_test, clf_temp.predict(X_test_reshaped)))

plt.figure(figsize=(10,6))
plt.plot(max_depth_list, train_accuracy, label='Train Accuracy', marker='o')
plt.plot(max_depth_list, test_accuracy, label='Test Accuracy', marker='o')
plt.xlabel('Profondeur (max_depth)')
plt.ylabel('Accuracy')
plt.title('Accuracy en fonction de max_depth')
plt.legend()
plt.grid(True)
plt.show()

param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4, 5] 
}

grid_search = GridSearchCV(SVC(random_state=0), param_grid, cv=5)
grid_search.fit(X_train_reshaped, y_train)

print(f"Meilleurs hyperparamètres : {grid_search.best_params_}")
print(f"Meilleure accuracy en validation croisée : {grid_search.best_score_:.2f}")

val_bike_folder = "val/bike"
val_car_folder = "val/car"

val_bike_images, val_bike_labels = populate_images_and_labels_lists(val_bike_folder, "bike")
val_car_images, val_car_labels = populate_images_and_labels_lists(val_car_folder, "car")

all_val_images = val_bike_images + val_car_images
all_val_labels = val_bike_labels + val_car_labels

val_images = np.array(all_val_images)
val_labels = np.array(all_val_labels)

best_max_depth = 6

clf_best = DecisionTreeClassifier(max_depth=best_max_depth, random_state=0)
clf_best.fit(X_train_reshaped, y_train)

val_images_reshaped = val_images.reshape(val_images.shape[0], -1)
val_pred = clf_best.predict(val_images_reshaped)

val_accuracy = accuracy_score(val_labels, val_pred)
print(f"Accuracy de validation : {val_accuracy:.2f}")

def populate_and_augment_images_and_labels_lists(image_folder_path, label):
    images = []
    labels = []
    
    for filename in os.listdir(image_folder_path):
        image = cv2.imread(os.path.join(image_folder_path, filename))
        
        resized_image = cv2.resize(image, target_size)
        
        cropped_image = resized_image[48:162, 48:162]
        cropped_image = cv2.resize(cropped_image, target_size)  
        
        grey_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        grey_image = cv2.cvtColor(grey_image, cv2.COLOR_GRAY2BGR)
        
        images.extend([resized_image, cropped_image, grey_image])
        labels.extend([label, label, label])
        
    return images, labels

augmented_bike_images, augmented_bike_labels = populate_and_augment_images_and_labels_lists(val_bike_folder, "bike")
augmented_car_images, augmented_car_labels = populate_and_augment_images_and_labels_lists(val_car_folder, "car")

all_augmented_images = augmented_bike_images + augmented_car_images
all_augmented_labels = augmented_bike_labels + augmented_car_labels

augmented_images_array = np.array(all_augmented_images)
augmented_labels_array = np.array(all_augmented_labels)

augmented_images_array_reshaped = augmented_images_array.reshape(augmented_images_array.shape[0], -1)

clf_best.fit(augmented_images_array_reshaped, augmented_labels_array)

val_pred_augmented = clf_best.predict(val_images_reshaped)
val_accuracy_augmented = accuracy_score(val_labels, val_pred_augmented)
print(f"Accuracy de validation après augmentation des données : {val_accuracy_augmented:.2f}")
