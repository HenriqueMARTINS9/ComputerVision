# ComputerVision
TP Computer Vision ESIEE-IT E5

## Mise en place de l’environnement de code

XI. A quelle fonction la distribution de noise fait penser ?  
La distribution de noise fait penser à une fonction gaussienne.

## Données

I. Combien y a-t-il d’images ? (On ne les compte pas à la main !)  
Il y a 916 images.

II. Quel est le format et la taille des images ?  
Toutes les images sont au format JPEG ou JPG et une image est en PNG mais les tailles sont toutes différentes les unes des autres.

III. A quoi sert l’argument random_state ?  
L'argument random_state est un nombre qui contrôle la façon dont le générateur pseudo-aléatoire divise les données. 

## Modèles de classification

I. Comment prédire le label de la première image du set de test ?  
Voir lignes 148 et 149.  

II. Interpréter la matrice de confusion : combien de bike ont été classifiés comme descar ? Combien de car ont été classifiés comme des bike ?  
10 bike classifiés comme car et 6 car classifiés comme bike.

## Comparaison de pipeline et fine tuning

I. Quelle est la meilleure valeur de max_depth à choisir ? Pourquoi ?  
Pour déterminer la meilleure valeur de max_depth, il faut trouver un équilibre entre la performance sur l'ensemble d'entraînement et l'ensemble de test, ici on pourrait dire que cette valeur est 6.

II. Que peut-on dire de cette valeur ?  
Si l'accuracy de validation est significativement inférieure à l'accuracy d'entraînement, cela pourrait indiquer un surapprentissage (overfitting) du modèle sur les données d'entraînement. Cela signifie que le modèle a mémorisé les données d'entraînement plutôt que d'apprendre les caractéristiques générales, ce qui le rend moins performant sur de nouvelles données.
Si l'accuracy de validation est similaire ou même meilleure que l'accuracy d'entraînement, cela indique que le modèle généralise bien les nouvelles données.

III. Comment peut-on l’expliquer ?  
Il y a plusieurs raisons possibles pour le surapprentissage : la complexité du modèle, la taille de l'ensemble de données, la variabilité des données, le prétraitement.
Pour lutter contre le surapprentissage, on peut essayer d'ajouter plus de données, d'utiliser la validation croisée, d'appliquer une régularisation ou d'essayer d'autres techniques de prétraitement.

IV. Quelle est la dimension d’une grey_image après la première ligne ? Quelle est ladimension d’une grey_image après la deuxième ligne ? Sur quel paramètre joue-t-on ?  
Après la première ligne, grey_image est une image en niveaux de gris avec les mêmes dimensions que resized_image, mais avec une seule channel (H, W). Donc, la dimension est (224, 224).

Après la deuxième ligne, grey_image est convertie en une image à trois channels (mais toujours en nuances de gris). La dimension est (224, 224, 3).

Le paramètre sur lequel on joue est le nombre de channels de l'image.

V. Quel est l’intérêt de cette deuxième ligne dans la transformation en noir et blanc ?  
La deuxième ligne convertit l'image en niveaux de gris en une image à trois channels. L'intérêt de cela est de s'assurer que toutes les images, qu'elles soient en couleur ou en niveaux de gris, ont le même nombre de channels. Cela est essentiel pour entraîner le modèle, car il attend une forme d'entrée cohérente.


