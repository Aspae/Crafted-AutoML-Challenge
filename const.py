from classesProblems import Classes

rand_state = 42
verbose = 2

EVAL_METRICS = {
    Classes.Regression: ["neg_root_mean_squared_error", "r2", "neg_mean_absolute_error"],
    Classes.Binary: ["roc_auc", "accuracy", "f1_macro"],
    Classes.Multi_class: ["f1_macro", "accuracy"],
    Classes.Multi_label: ["f1_macro", "jaccard_samples"]
}