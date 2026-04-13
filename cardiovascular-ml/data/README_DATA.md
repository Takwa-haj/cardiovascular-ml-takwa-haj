# Data Folder

This folder contains the datasets used in the project.

## Automatic download

```bash
python data/download_data.py
```

## Datasets

| File | Source | Size | Download |
|------|--------|------|----------|
| `uci_heart.csv` | UCI ML Repository | ~30 KB | Auto (script above) |
| `framingham.csv` | Kaggle / NHLBI | ~500 KB | [Manual](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset) |
| `cardio_train.csv` | Kaggle (Sulianova) | ~10 MB | [Manual](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) |

> CSV files are excluded from git via `.gitignore`. Run the download script after cloning.
