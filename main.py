import automl

def main():
    data_dest = "/info/corpus/ChallengeMachineLearning"
    automl.fit(data_dest)
    automl.eval()

if __name__ == "__main__":
    main()