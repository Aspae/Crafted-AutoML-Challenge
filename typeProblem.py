from classesProblems import Classes

def getTypeProblem(dataSolution):
    numberOfOutput = dataSolution.shape[1]
    
    if numberOfOutput == 1:
        nbValsTot = dataSolution.shape[0]
        uniqVals = len(dataSolution[0].unique())

        # Régression
        if (uniqVals / nbValsTot) > 0.30:
            return Classes.Regression
        
        elif uniqVals == 2:
            return Classes.Binary
        
        else:
            return Classes.Multi_class
    
    else:
        return Classes.Multi_label
        

