import pandas as pd
import numpy as np
import os


DATA_ROOT_PATH = "/info/corpus/ChallengeMachineLearning"


def getDataRaw(dataSetName: str):
    dataRaw = pd.read_csv(
        DATA_ROOT_PATH + f"/data_{dataSetName}/data_{dataSetName}.data",
        sep=r"\s+",
        header=None,
    )

    # Suppression de la dernière colonne si elle contient uniquement des NaN
    if dataRaw.iloc[:, -1].isna().all():
        dataRaw.drop(columns=dataRaw.columns[-1], inplace=True)
    
    return dataRaw
    



def getdataType(dataSetName: str):
    dataType = pd.read_csv(
        DATA_ROOT_PATH + f"/data_{dataSetName}/data_{dataSetName}.type",
        header=None,
    )
    
    # Suppression de la dernière colonne si elle contient uniquement des NaN
    if dataType.iloc[:, -1].isna().all():
        dataType.drop(columns=dataType.columns[-1], inplace=True)
    
    return dataType




def getDataSolution(dataSetName: str):
    dataSolution = pd.read_csv(
        DATA_ROOT_PATH + f"/data_{dataSetName}/data_{dataSetName}.solution",
        sep=r"\s+",
        header=None,
    )

    # Suppression de la dernière colonne si elle contient uniquement des NaN
    if dataSolution.iloc[:, -1].isna().all():
        dataSolution.drop(columns=dataSolution.columns[-1], inplace=True)
    

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

    return dataSolution
        