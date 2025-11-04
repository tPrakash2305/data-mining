import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess_data(path=None):
    """Load Iris CSV and return scaled, split data.

    If `path` is None, the function will look for `data/Iris.csv` in the
    current working directory. If `path` is a directory, it will try to read
    `Iris.csv` from that directory.
    """
    if path is None:
        path = os.path.join(os.getcwd(), "data", "Iris.csv")

    # If user passed a directory, attempt to use Iris.csv inside it
    if os.path.isdir(path):
        path = os.path.join(path, "Iris.csv")

    df = pd.read_csv(path)

    # Drop unnecessary column if present
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    X = df.drop(columns=["Species"])
    y = df["Species"]

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    return X_train, X_test, y_train, y_test, le
