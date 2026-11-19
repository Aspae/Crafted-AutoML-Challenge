import numpy as np
import pandas as pd

def deleteNegEstimator(scores):
    cleaned = {}
    for key, val in scores.items():
        if key.startswith("neg_"):
            cleaned[key[4:]] = -val
        else:
            cleaned[key] = val
    return cleaned

def save_predictions(preds, path):
    if isinstance(preds, np.ndarray):
        arr = np.asarray(preds)

        if arr.ndim == 1:
            pd.DataFrame({"prediction": arr}).to_csv(path, index=False)
        elif arr.ndim == 2:
            df = pd.DataFrame(arr)
            df.columns = [f"prediction_{i + 1}" for i in range(df.shape[1])]
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"Prediction shape unsupported for saving: {arr.shape}")
    else:
        preds.to_csv(path, index=False)
