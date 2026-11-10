import dataExtract
import preProcessor
from modelSelection import best_model, eval_model
import typeProblem
from const import EVAL_METRICS

class AutoML:

    def __init__(self):
        self.model = None
        self.report = None
        self.test_scores = None
        self.type_problem = None

    def fit(self, data_dest: str):
        data = "C"
        X = dataExtract.getDataRaw(data)
        y = dataExtract.getDataSolution(data)
        self.type_problem = typeProblem.getTypeProblem(y)
        print(f"Type of problem identified : {self.type_problem}")

        X_train, X_val, X_test, y_train, y_val, y_test = preProcessor.train_val_test_split(X, y)

        self.model, self.report = best_model(self.type_problem, X_train, X_val, y_train, y_val)

        self.test_scores = eval_model(self.model, X_test, y_test, EVAL_METRICS[self.type_problem])

    def eval(self):
        for rank, entry in enumerate(self.report, start=1):
            print(rank, entry["name"], "| cv:", round(entry["cv_score"], 4), "| val:", entry["val_scores"])
        print("test:", self.test_scores)

    def predict(self, data_dest: str):
        pass

_instance = AutoML()

fit = _instance.fit
eval = _instance.eval
predict = _instance.predict