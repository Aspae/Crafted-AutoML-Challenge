import pandas as pd
import numpy as np
import os


DATA_ROOT_PATH = "/info/corpus/ChallengeMachineLearning"


def getDataRaw(dataSetName: str):
    dataRaw = pd.read_csv(
        DATA_ROOT_PATH + f"/data_{dataSetName}/data_{dataSetName}.data",
        sep=" ",
        header=None,
    )
    dataRaw.drop(columns=dataRaw.columns[-1], inplace=True) # suppression de la dernière colonne à NaN
    return dataRaw
    

def getdataType(dataSetName: str):
    dataType = pd.read_csv(
        DATA_ROOT_PATH + f"/data_{dataSetName}/data_{dataSetName}.type",
        header=None,
    )
    return dataType


def getDataSolution(dataSetName: str):
    dataSolution = pd.read_csv(
        DATA_ROOT_PATH + f"/data_{dataSetName}/data_{dataSetName}.solution",
        sep=" ",
        header=None,
    )
    dataSolution.drop(columns=dataSolution.columns[-1], inplace=True) # suppression de la dernière colonne à NaN
    return dataSolution
        

dataRaw = getDataRaw("A")
dataType = getdataType("A")
dataSolution = getDataSolution("A")
print(dataRaw[:5])
print(dataType[:5])
print(dataSolution[:5])


dataRaw = getDataRaw("B")
dataType = getdataType("B")
dataSolution = getDataSolution("B")
print(dataRaw[:5])
print(dataType[:5])
print(dataSolution[:5])


dataRaw = getDataRaw("C")
dataType = getdataType("C")
dataSolution = getDataSolution("C")
print(dataRaw[:5])
print(dataType[:5])
print(dataSolution[:5])

