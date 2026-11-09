from sklearn.model_selection import train_test_split

from const import rand_state

def train_val_test_split(X, y):
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.30, random_state=rand_state)

    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=rand_state)

    return X_train, X_val, X_test, y_train, y_val, y_test