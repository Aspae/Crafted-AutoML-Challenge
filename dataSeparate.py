from sklearn.model_selection import train_test_split

def separateData(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test