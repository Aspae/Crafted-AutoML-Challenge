import pandas as pd
import numpy as np
import os
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix


DATA_ROOT_PATH = "/info/corpus/ChallengeMachineLearning"

# vérifie si les données sources représentent une matrice creuse
def dataIsSparse(datasetName: str):
    limit = 100
    
    with open(DATA_ROOT_PATH + f"/data_{datasetName}/data_{datasetName}.data") as f:
        for ligne in f:
            for element in ligne.split():
                if ":" not in element:
                    return False
    
    return True


# chargement d'une matrice de type creuse
def getDataRawFromSparse(datasetName: str):
    data = []
    indices = []
    indptr = [0]

    with open(DATA_ROOT_PATH + f"/data_{datasetName}/data_{datasetName}.data") as f:
        for ligne in f:
            for element in ligne.split():
                indice, valeur = element.split(":")
                indices.append(int(indice) - 1)
                data.append(float(valeur))

            indptr.append(len(data))

    dataRaw = csr_matrix(
        (data, indices, indptr),
        shape=(len(indptr) - 1, max(indices) + 1)
    )

    return dataRaw


# chargement des données sources
def getDataRaw(datasetName: str):
    
    # si la matrice est creuse on utilise la bonne fonction pour charger les données
    if dataIsSparse(datasetName):
        return getDataRawFromSparse(datasetName)
    
    # sinon on fait le chargement classique
    dataRaw = pd.read_csv(
        DATA_ROOT_PATH + f"/data_{datasetName}/data_{datasetName}.data",
        sep=r"\s+",
        header=None,
    )

    # Suppression de la dernière colonne si elle contient uniquement des NaN
    if dataRaw.iloc[:, -1].isna().all():
        dataRaw.drop(columns=dataRaw.columns[-1], inplace=True)
    
    return dataRaw


# chargement du type de données
def getdataType(datasetName: str):
    dataType = pd.read_csv(
        DATA_ROOT_PATH + f"/data_{datasetName}/data_{datasetName}.type",
        header=None,
    )
    
    # Suppression de la dernière colonne si elle contient uniquement des NaN
    if dataType.iloc[:, -1].isna().all():
        dataType.drop(columns=dataType.columns[-1], inplace=True)
    
    return dataType


# chargement des labels attendus
def getDataSolution(datasetName: str):
    dataSolution = pd.read_csv(
        DATA_ROOT_PATH + f"/data_{datasetName}/data_{datasetName}.solution",
        sep=r"\s+",
        header=None,
    )

    # Suppression de la dernière colonne si elle contient uniquement des NaN
    if dataSolution.iloc[:, -1].isna().all():
        dataSolution.drop(columns=dataSolution.columns[-1], inplace=True)
    
    '''
    # Si la classe solution est représentée par "0 0 0 0 0 0 0 1 0 0" on revoie la valeur de l'index (ici 7)
    sumLines = dataSolution.sum(axis=1)
    uniqValTab = sumLines.unique()
    if len(uniqValTab) == 1:
        uniqVal = uniqValTab[0]

        # Transformation en DataFrame
        indices = pd.DataFrame(
            dataSolution.eq(uniqVal).values.argmax(axis=1).reshape(-1, 1),
            index=dataSolution.index
        )
        dataSolution = indices
    '''
    
    return dataSolution
        