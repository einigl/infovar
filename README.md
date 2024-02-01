# Infovar

Dans ce projet, nous proposons de mesurer l'informativité des variables en nous servant de trois statistiques :
- l'entropie conditionelle $h$
- l'information mutuelle $I$
- le coefficient de corrélation canonique $\rho$

Les deux premières sont estimées de manière non-paramétrique. Nous utiliserons l'implémentation de Greg Ver Steeg (voir: http://www.isi.edu/~gregv/npeet.html). Le coefficient de corrélation canonique est calculé comme la première valeur propre de la matrice de contraction des données. Plus de détails sont donnés en section __Statistics__.

Si d'autres métriques vous intéressent, il est possible ajouter et de les utiliser.


### Informativity

L'informativité d'une variable aléatoire $X$ sur une variable aléatoire $Y$ est la capacité de la première à contraindre les valeurs possibles de la seconde. 

Cette informativité peut résulter en une capacité à être utilisée en régression (c'est-dire-chercher une function $f$ telle que $f(X) \approx Y$), mais ce n'est pas une condition nécessaire.


### Statistics

#### Conditional entropy

__Definition:__

TODO

__Properties:__

TODO

__Estimation:__

TODO

#### Mutual information

__Definition:__

TODO

__Properties:__

TODO

__Estimation:__

TODO

#### Canonical correlation coefficient

__Definition:__

TODO

__Properties:__

TODO

__Estimation:__

Directly from definition.

### Uncertainty on estimations

L'incertitude sur l'estimation des statistiques ci-dessus peut résulter de différentes sources:
- la variance de l'estimateur
- les fluctuations statistiques d'échantillons issues d'une même distribution

Pour prendre en compte ces incertitudes et pouvoir comparer proprement différentes valeurs, nous proposons de nous baser sur une approche dites de "bootstrapping", qui consiste à tirer avec remplacement des échantillons à partir d'un même jeu de données.
