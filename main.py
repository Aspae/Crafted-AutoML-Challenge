import dataExtract as De
import dataSeparate as Ds

def main():
    Xa = De.getDataRaw('A')
    ya = De.getDataSolution('A')

    X_train, X_val, X_test, y_train, y_val, y_test = Ds.separateData(Xa, ya)

    print(f"Nb data: {len(Xa)}\n")
    print(f"Nb train: {len(X_train)}\n")
    print(f"Nb val: {len(X_val)}\n")
    print(f"Nb test: {len(X_test)}")

if __name__ == "__main__":
    main()