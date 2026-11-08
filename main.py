import typeProblem
import dataExtract

def main():
    dataA = dataExtract.getDataSolution("A")
    dataB = dataExtract.getDataSolution("B")
    dataC = dataExtract.getDataSolution("C")
    dataD = dataExtract.getDataSolution("D")
    dataE = dataExtract.getDataSolution("E")
    dataF = dataExtract.getDataSolution("F")
    
    print(typeProblem.getTypeProblem(dataA))
    print(typeProblem.getTypeProblem(dataB))
    print(typeProblem.getTypeProblem(dataC))
    print(typeProblem.getTypeProblem(dataD))
    print(typeProblem.getTypeProblem(dataE))
    print(typeProblem.getTypeProblem(dataF))
if __name__ == "__main__":
    main()