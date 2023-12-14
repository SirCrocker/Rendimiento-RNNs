import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

VALID_LABELS = ["yes", "no", "up", "down", "left", "right", "on",
"off", "stop", "go", "UnkwnW"]

LABELS = ["down", "go", "left", "no", "off", "on", "right", "stop", "UnkwnW", "up", "yes"]

def Wav2Vec_CM():
    df = pd.read_csv("ConfusionMatrix/Wav2VecDataProc.csv")
    df_models = [y for _, y in df.groupby("name")]

    for model in df_models:

        model_name = model["name"].iloc[0]
        cm = model.pivot(index="Actual", columns="Predicted", values="nPredictions")
        normalized_df=cm/cm.sum(axis=1)
        
        fig, ax = plt.subplots(figsize=(7,6))
        sns.heatmap(normalized_df, annot=True, fmt='.2f',
                    xticklabels=LABELS, yticklabels=LABELS,
                    cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
        plt.title("Confusion Matrix for " + model_name + " with Wav2Vec")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f"ConfusionMatrix/imgs/{model_name}-W2V.pdf", dpi=500)
    return

def MFCC_CM():
    df = pd.read_csv("ConfusionMatrix/MFCCDataProc.csv")
    df_models = [y for _, y in df.groupby("name")]
    for model in df_models:

        model_name = model["name"].iloc[0]
        cm = model.pivot(index="Actual", columns="Predicted", values="nPredictions")
        normalized_df=cm/cm.sum(axis=1)

        if model_name == "RNNModel":
            print(model[model["Actual"] == "right"])
            print(cm)
        
        fig, ax = plt.subplots(figsize=(7,6))
        sns.heatmap(normalized_df, annot=True, fmt='.2f',
                    xticklabels=LABELS, yticklabels=LABELS,
                    cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
        plt.title("Confusion Matrix for " + model_name + " with MFCC")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f"ConfusionMatrix/imgs/{model_name}-MFCC.pdf", dpi=500)
        plt.close()
    return

if __name__ == "__main__":
    Wav2Vec_CM()
    print("Done CM Wav2Vec")
    MFCC_CM()
    print("Done CM MFCC")