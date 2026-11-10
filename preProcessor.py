import pandas as pd
import numpy as np

'''
Ce code a pour but de normaliser les données affin qu'elles passent mieux dans le modèle
'''


'''
Binary
Categorical

Numerical
'''




def NaN_replace(data, dataType):
    for i, col in enumerate(data.columns.copy()):
        typeColonne = dataType.iloc[i, 0]

        # colonne avec que des NaN
        if not data[col].notna().any():
            data = data.drop(columns=[col])
            dataType = dataType.drop(columns=[col])
            print(f"{i} : COL NaN")

        # on remplace les valeurs manquantes par la médiane
        elif typeColonne == "Numerical":
            data[col] = data[col].fillna(data[col].median())
            print(f"{i} : MEDIANE")

        # on remplace les valeurs manquantes par la classe majoritaire
        elif typeColonne in ["Binary", "Categorical"]:
            data[col] = data[col].fillna(data[col].mode()[0])
            print(f"{i} : CLASSE MAJ")

    return data, dataType



def delete_col_variance_at_0(data, dataType):
    for i, col in enumerate(data.columns.copy()):
        variance = data[col].var()

        # colonne avec variance = 0
        if variance == 0:
            data = data.drop(columns=[col])
            dataType = dataType.drop(columns=[col])
            print(f"{i} : COL avec variance = 0")

    return data, dataType


from const import rand_state

def train_val_test_split(X, y):
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.30, random_state=rand_state)

    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=rand_state)

    return X_train, X_val, X_test, y_train, y_val, y_test
def delete_outliers_lines(data, dataType, dataSolution, borneInf=0.25, borneSup=0.75):
    listeOutliers = []
    for i, col in enumerate(data.columns.copy()):

        # si la colonne n'est pas numérique on ne fais pas de traitements
        typeColonne = dataType.iloc[i, 0]
        if typeColonne != "Numerical":
            continue

        Q1 = data[col].quantile(borneInf)
        Q3 = data[col].quantile(borneSup)

        ecartInterquartile = Q3 - Q1

        borne_min = Q1 - 1.5 * ecartInterquartile
        borne_max = Q3 + 1.5 * ecartInterquartile

        outliers = data[(data[col] < borne_min) | (data[col] > borne_max)]

        nbOutliers = len(outliers)
        print(f"{col} : Nombre d'outliers = {nbOutliers}")

        listeOutliers.append(nbOutliers)
        if len(listeOutliers) > 100:
            listeOutliers.pop(0)

            if all(x == 0 for x in listeOutliers):
                break

        data = data.drop(index=outliers.index)

    dataType = dataType.loc[data.columns]
    dataSolution = dataSolution.loc[data.index]

    return data, dataType, dataSolution


"""
aux colonnes numériques dont le coefficient d'asymétrie (skew) est supérieur à 1
"""
def log_transform(data, dataType, seuil=1):
    for i, col in enumerate(data.columns):

        typeColonne = dataType.iloc[i, 0]

        if typeColonne == "Numerical":
            asymetrie = data[col].skew()

            if asymetrie > seuil:
                data[col] = np.log1p(data[col])
                print(f"{col} : LOG (skew = {asymetrie:.2f})")

    return data, dataType