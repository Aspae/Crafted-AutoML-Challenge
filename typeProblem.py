from classesProblems import Classes

import numpy as np
def getTypeProblem(dataSolution):
    if dataSolution.ndim == 1:
        numberOfOutput = 1
        
    else:
        numberOfOutput = dataSolution.shape[1]
    
    if numberOfOutput == 1:
        nbValsTot = dataSolution.shape[0]
        uniqVals = len(np.unique(dataSolution))

        # Régression
        if (uniqVals / nbValsTot) > 0.30:
            return Classes.Regression
        
        elif uniqVals == 2:
            return Classes.Binary
        
        else:
            return Classes.Multi_class
    
    else:
        return Classes.Multi_label
        

