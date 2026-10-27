"""
Preparación de datos para el proyecto Telco Customer Churn (MLY1101 - EP2).

Principio anti-leakage
----------------------
- `clean_and_engineer` aplica solo reglas deterministas fila a fila (no aprende nada de los
  datos), por lo que puede ejecutarse ANTES del split sin filtrar información del test.
- Todo lo que "aprende" de los datos (escalado, one-hot, ordinal) vive en el
  ColumnTransformer de `build_preprocessor`, que se ajusta SOLO con el conjunto de train
  (y, en la validación cruzada, solo con el fold de entrenamiento al ir dentro de un Pipeline).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET = "Churn"

# Servicios (tras unificar "No internet service"/"No phone service" a "No")
SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
PROTECTION_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
YES_NO_COLS = ["Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity",
               "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "PaperlessBilling"]

TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-72"]
CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]

# Definición de columnas por tipo de transformación
NUM_COLS = ["tenure", "MonthlyCharges", "log_TotalCharges", "charge_ratio",
            "n_services", "n_protection"]
ORD_COLS = ["Contract", "tenure_group"]
OHE_COLS = ["InternetService", "PaymentMethod"]
BIN_COLS = ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "PaperlessBilling", "auto_payment",
            "fiber_monthly"]
FEATURES = NUM_COLS + ORD_COLS + OHE_COLS + BIN_COLS   # `gender` y `customerID` se excluyen a propósito


def clean_and_engineer(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Limpieza + variables derivadas. Reglas deterministas (sin aprendizaje => sin leakage)."""
    df = df_raw.copy()

    # 1) TotalCharges: texto -> numérico. Los vacíos son clientes con tenure = 0 => 0 (no mediana)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0

    # 2) Unificar categorías redundantes a "No"
    df = df.replace({"No internet service": "No", "No phone service": "No"})

    # 3) Variable objetivo y binarias Yes/No -> 1/0
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    for c in YES_NO_COLS:
        df[c] = (df[c] == "Yes").astype(int)

    # 4) Variables derivadas
    df["log_TotalCharges"] = np.log1p(df["TotalCharges"])
    avg_hist = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"].replace(0, np.nan),
                        df["MonthlyCharges"])
    df["charge_ratio"] = df["MonthlyCharges"] / avg_hist          # >1: paga hoy más que su promedio histórico
    df["tenure_group"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    df["n_services"] = df[SERVICE_COLS].sum(axis=1)
    df["n_protection"] = df[PROTECTION_COLS].sum(axis=1)
    df["auto_payment"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    df["fiber_monthly"] = ((df["InternetService"] == "Fiber optic") &
                           (df["Contract"] == "Month-to-month")).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer sin ajustar. Usar .fit() solo con train (o dentro de un Pipeline)."""
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("ord", OrdinalEncoder(categories=[CONTRACT_ORDER, TENURE_LABELS]), ORD_COLS),
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), OHE_COLS),
            ("bin", "passthrough", BIN_COLS),
        ],
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")
    return pre
