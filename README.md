<h2 align="center">FlightRank 2025: Aeroclub RecSys Cup — 4th Solution</h2>

This repository contains the source code for our best XGBoost model, achieving a **LB of 0.52795** and **PB of 0.53552**.

### Kaggle Write-Up
[4th Solution: XGBoost Model](https://www.kaggle.com/competitions/aeroclub-recsys-2025/writeups/4th-solution-xgboost-model)

### File Structure

```txt
├── extract.ipynb       # Extract additional information from raw JSON files
├── flight.ipynb        # Main notebook
├── requirements.txt    # Python dependencies
├── run.sh              # Script to run notebook in the background
└── src
    ├── __init__.py
    ├── data.py             # Data splitting
    ├── feature.py          # Feature engineering and selection
    ├── feature_specs.py    # Feature specifications used in the model
    ├── params.py           # Model hyperparameters
    ├── plot.py             # Visualization functions
    └── utils.py            # Utility functions: evaluation, reranking, prediction
```

### How to Run the Code

1. Download the data from [Kaggle Aeroclub RecSys 2025 Data](https://www.kaggle.com/competitions/aeroclub-recsys-2025/data) and place it in the `./data/` directory. Follow the instructions to unzip `jsons_raw.tar.kaggle`.
2. Create a Conda environment and install the required packages from `requirements.txt`:
    ```bash
    conda create -n FlightRank python=3.10
    conda activate FlightRank
    pip install -r requirements.txt
    ```
3. Run `extract.ipynb` to extract additional information from the raw JSON files.
4. Configure `flight.ipynb`:
   - `FULL = True` — train on the full training dataset.
   - `FULL = False` — train on 90% of the training data for local validation.
5. Optionally, adjust the training-validation split sizes in `utils.py`:
   - `TRAIN_VAL_SIZE` — size of the training-validation split.
   - `TRAIN_ALL_SIZE` — size of the dataset when modifying the training set.
6. To add more features, modify or add functions in `feature.py` and include them in the `feature_engineering` workflow.
7. For feature selection, update the `FeatureSpec` class in `feature_specs.py` by adding or removing features as needed.
