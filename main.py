import automl
import utils

def main():
    data_dest = "/info/corpus/ChallengeMachineLearning"
    dataID = "A"
    data_dest += f"/data_{dataID}/data_{dataID}"
    automl.fit(data_dest)
    automl.eval()
    preds = automl.predict(data_dest)

    # créer un fichier predictions.txt pour vérifier que le format de retour est le bon
    utils.save_predictions(preds, "predictions.txt")

if __name__ == "__main__":
    main()   