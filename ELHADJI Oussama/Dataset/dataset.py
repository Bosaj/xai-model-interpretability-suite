import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
try:
    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH
except ImportError:
    CS = None
    CSH = None

try:
    import torch
    from torch.utils.data import Dataset as PyDataset
except ImportError:
    torch = None
    PyDataset = object

LABELS = {
    "bike_sharing": [
        "Instant", "Date", "Season", "Year", "Month", "Holiday", "Weekday", "Working day",
        "Weather situation", "Temperature", "Feeling Temperature", "Humidity", "Windspeed",
        "Casual", "Registered", "Shared Bikes"
    ],
    "ph": ["Blue", "Green", "Red", "pH"],
    "cancer": [
        "Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies",
        "Smoking", "Smoking (years)", "Smokes (packs/year)", "Hormonal Contraceptives",
        "Hormonal Contraceptives (years)", "Intrauterine Device", "Intrauterine Device (years)",
        "Sexually Transmitted Disease", "Sexually Transmitted Disease (number)",
        "STDs:condylomatosis", "STDs:cervical condylomatosis", "STDs:vaginal condylomatosis",
        "STDs:vulvo-perineal condylomatosis", "STDs:syphilis", "STDs:pelvic inflammatory disease",
        "STDs:genital herpes", "STDs:molluscum contagiosum", "STDs:AIDS", "STDs:HIV",
        "STDs:Hepatitis B", "STDs:HPV", "STDs: Number of diagnosis", "STDs: Time since first diagnosis",
        "Sexually Transmitted Disease (Time since last diagnosis)", "Dx:Cancer", "Dx:CIN",
        "Dx:HPV", "Dx", "Hinselmann", "Schiller", "Citology", "Biopsy"
    ],
    "water_potability": [
        "pH", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity",
        "Organic Carbon", "Trihalomethanes", "Turbidity", "Potability"
    ],
    "wheat_seeds": [
        "area", "perimeter", "compactness", "length", "width", "asymmetry", "groove", "type"
    ],
    "credit": [
        "credit_risk", "age", "amount", "credit_history", "duration",
        "employment_duration", "personal_status_sex", "purpose"
    ],
    "fifa": [
        "Date", "Team", "Opponent", "Goal Scored", "Ball Possession %", "Attempts",
        "On-Target", "Off-Target", "Blocked", "Corners", "Offsides", "Free Kicks", "Saves",
        "Pass Accuracy %", "Passes", "Distance Covered (Kms)", "Fouls Committed", "Yellow Card",
        "Yellow & Red", "Red", "Man of the Match", "1st Goal", "Round", "PSO",
        "Goals in PSO", "Own goals", "Own goal Time"
    ]
}


class Dataset:
    def __init__(self,
                 dataset_name,
                 input_ids,
                 output_id,
                 normalize: bool = False,
                 categorical: bool = False,
                 impute_strategy: str = "remove"):
        
        # C'EST ICI QUE LA MAGIE OPÈRE POUR VOTRE DOSSIER "Dataset"
        possible_paths = [
            os.path.join("datasets", dataset_name + ".csv"),
            dataset_name + ".csv",
            os.path.join(os.path.dirname(__file__), dataset_name + ".csv"),
            os.path.join("Dataset", dataset_name + ".csv")  # <--- Cette ligne trouve votre dossier
        ]
        
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
                
        if csv_path is None:
            csv_path = dataset_name + ".csv"
            
        self._data = pd.read_csv(csv_path).to_numpy()

        self.dataset_name = dataset_name
        self.input_ids = input_ids

        if isinstance(output_id, list) and len(output_id) == 1:
            output_id = output_id[0]

        self.output_id = output_id

        X, y = self._data[:, self.input_ids], self._data[:,
                                                         self.output_id].reshape(-1, 1)

        if impute_strategy is not None:
            if impute_strategy == "zeros":
                X[pd.isnull(X)] = 0
            elif impute_strategy == "remove":
                mask = np.any(pd.isnull(X), axis=1)
                X = X[~mask]
                y = y[~mask]
                mask = np.any(pd.isnull(y), axis=1)
                X = X[~mask]
                y = y[~mask]
            else:
                raise NotImplementedError("Impute strategy was not found.")

        if normalize:
            scaler = MinMaxScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            if not categorical:
                scaler.fit(y)
                y = scaler.transform(y)

        self.X = X
        self.y = y

    def get_configspace(self):
        if CS is None or CSH is None:
            raise ImportError("ConfigSpace module is required. Install with 'pip install ConfigSpace'.")
        cs = CS.ConfigurationSpace()
        for j in range(self.X.shape[1]):
            name = self.get_input_labels(j)
            uniform_float_hp = CSH.UniformFloatHyperparameter(
                name,
                lower=np.min(self.X[:, j]),
                upper=np.max(self.X[:, j]),
                log=False)
            cs.add_hyperparameter(uniform_float_hp)
        return cs

    def get_data(self, split=0.6, random_state=0):
        X, y = shuffle(self.X, self.y, random_state=random_state)
        split_idx = int(len(X)*split)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]
        y_train = y_train.flatten()
        y_val = y_val.flatten()
        return (X_train, y_train), (X_val, y_val)

    def get_input_labels(self, id=None):
        l = np.array(LABELS[self.dataset_name])
        if id is None:
            return l[self.input_ids]
        else:
            return str(l[self.input_ids][id])

    def get_classes(self):
        return list(np.unique(self.y))

    def get_output_label(self):
        return LABELS[self.dataset_name][self.output_id]


class PyTorchDataset(PyDataset):
    """
    Since we have numpy data, it is required to convert
    them into PyTorch tensors first.
    """

    def __init__(self, X, y=None):
        if torch is None:
             raise ImportError("PyTorch is not installed. Please install it with 'pip install torch'.")
             
        self.X = torch.tensor(X, dtype=torch.float32)

        self.y = torch.zeros((self.X.shape[0], 1), dtype=torch.float32)
        if y is not None:
            y = y.astype(np.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__ == "__main__":
    try:
        dataset = Dataset("wheat_seeds", [1, 2, 3, 4], [0])
        (X_train, y_train), (X_test, y_test) = dataset.get_data()
        print("Dataset loaded successfully")
    except Exception as e:
        print(f"Error loading dataset: {e}")


