def getTypeProblem(dataSolution):
    numberOfOutput = dataSolution.shape[1]
    
    if numberOfOutput == 1:
        nbValsTot = dataSolution.shape[0]
        uniqVals = len(dataSolution[0].unique())

        # Régression
        if (uniqVals / nbValsTot) > 0.30:
            print("Régression")
        
        elif uniqVals == 2:
            print("Binaire")
        
        else:
            print("Multi-classes")
    
    else:
        print("Multi-labels")
        

