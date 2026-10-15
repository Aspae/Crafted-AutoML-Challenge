from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

"""
Pour la régression classique:
    - linear_model.LinearRegression
    - linear_model.Ridge
    - linear_model.Lasso
    - tree.DecisionTreeRegressor
    - ensemble.RandomForestRegressor
    - ensemble.HistGradientBoostingRegressor
    
    - neg_mean_squared_error ou r2 ou RMSE
    
Pour la classification binaire et multi-classe:
    - linear_model.LogisticRegression
    - svm.SVC
    - neighbors.KNeighborsClassifier
    - naive_bayes.GaussianNB
    - ensemble.RandomForestClassifier
    - ensemble.HistGradientBoostingClassifier
    
    - accuracy ou roc_auc => binaire
    - accuracy ou f1_macro => multi-class

Pour la classification multi-label:
    - multiclass.BinaryRelevance
    - multiclass.ClassifierChain
    - tree.DecisionTreeClassifier
    - ensemble.RandomForestClassifier
    - ClassifierChain(LogisticRegression())
    
    - f1_macro ou jaccard_score
"""