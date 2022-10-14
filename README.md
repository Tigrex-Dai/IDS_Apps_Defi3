# Défi 3 - classification

## Présentation

L'analyse pharmaceutique (AP) est une étape essentielle de la prévention des risques liés à la prescription des médicaments.

Au cours de l'AP, les pharmaciens identifient les problèmes liés aux prescriptions médicamenteuses (DRP) et déclenchent des interventions pharmaceutiques (IP) pour corriger les éventuels problèmes liés aux prescriptions.  


L'objectif du défi 2 est de *développer et valider un modèle de classification* pour catégoriser automatiquement les IP selon la classification de la Société française de pharmacie clinique.


Les données sont disponibles dans le fichier *data_defi3.csv.gz*

Ces données proviennent d'un travail réalisé aux Hôpitaux Universitaires de Strasbourg (en cours de publication) sur les métadonnées du logiciel d'aide à la prescription utilisé au sein de l'hôpital.

Les données sont composées de 3 colonnes indicant :

- le libéllé de la molécule prescrite

- le commentaire du pharmacien sur cette prescrition (déclanchant l'IP)

- la classe de l'IP (selon la classification  de la SFPC proposée dans le fichier SFPC_encodage.xlsx - encodage réalisé par 2 pharmaciens experts)


En résumé :

- dans un premier temps (quest 1) - vous devez développer un modèle permettant de prédire si une erreur de prescription potentiellement grave a été identifiée à partir des commentaires (une erreur est considérée comme grave si l'IP appartient aux classes 4 (Surdosage), 5 (médicament non indiqué) , 6.3 ou 6.4 (Interaction médicamenteuse - association déconseillée ou Contre_Indication) . (40 % du résultat)

- dans un second temps (quest 2) - vous devez développer un modèle permettant de classer automatiquement les commentaires selon les 11 classes principales (60% du résultat)


Votre rendu correspond à 2 scripts qui peuvent fonctionner sur un fichier de même structure que le fichier de données initial et qui retournent :

- le statut grave ou non encodé en 1/0 (quest 1).

- la classe du problème médicamenteux encodé de 1 à 11 (quest 2)

Vous aurez également à rendre les prédictions du jeu de données de validations (2 colonnes)

Un bonus sera attribué au groupe qui obtient la meilleure performance sur l'ensemble des deux scripts (exactitude, puis VPP (rappel) en cas d'égalité) sur le set de validation


## Echéances

- date de retour des scripts le 20/11/2022 à 18h  (à déposer sur moodle : remise de devoirs (fichier .r, .py ou .ipynb) et des encodages du jeu de validation

- présentations orales (et dévoilement du classement) : 24/11/2022



