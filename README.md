# PawDetector — Cat & Dog Image Recognition

End-to-end project: data audit -> cleaning -> preprocessing -> traditional ML (LR, KNN, SVM, RF, HOG, PCA)
-> custom CNNs -> augmentation -> transfer learning (MobileNetV2) -> evaluation -> Streamlit app.

## Structure
data/{raw,cleaned,processed}  notebooks/  src/  models/  app/app.py  results/  requirements.txt

## Reproduce (Google Colab)
1. Open the notebook, set Runtime -> GPU, set `SOURCE_DIR` to the dataset folder (Cat/ and Dog/ sub-folders).
2. Run all cells (seed = 42). Outputs: `results/model_comparison.csv`, `models/final_model.keras`, plots, logs.

## Run the app locally

In PowerShell, from the project folder:

```
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Open http://localhost:8501, upload a JPG/PNG, and read the prediction, probability and confidence.
`models/final_model.keras` must exist (created by the notebook).
On first launch, the app also downloads ImageNet weights for a separate classifier that rejects images
that are not recognized as cats or dogs.

## Notes
* Preprocessing (RGB conversion, 128x128 bilinear resize) lives in `src/preprocessing.py` and is shared by training and the app;
  the model itself rescales pixels internally.
* Predictions below the sidebar confidence threshold are shown as "Uncertain".
