import logging
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log=logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

RAW_FILES={"cleveland":"processed.cleveland.data", "hungarian":"processed.hungarian.data", "switzerland":"processed.switzerland.data","va":"processed.va.data"}
COLUMN_NAMES=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

DROP_COLS=["ca", "thal","slope","source"]
TARGET_COL="target"

CONTINUOUS_COLS=["age", "trestbps", "chol", "thalach", "oldpeak"]

CATEGORICAL_COLS=["sex", "cp", "fbs", "restecg", "exang"]

FEATURE_COLS=CONTINUOUS_COLS+CATEGORICAL_COLS

def load_data(path:Path=DATA_DIR) -> tuple[pd.DataFrame, pd.Series]:
    log.info("Loading data from %s", path)
    frames=[]
    for name,filename in RAW_FILES.items():
        file_path=path/filename
        if not file_path.exists():
            log.warning("File %s does not exist", file_path)
            continue
        df_part=pd.read_csv(file_path, header=None, names=COLUMN_NAMES, na_values="?")
        df_part["source"]=name
        frames.append(df_part)
        log.info("Loaded %s with shape %s", name, df_part.shape)
            
    if not frames:
        log.error("No data files loaded. Please check the data directory.")
        raise FileNotFoundError("No data files found in the specified directory.")
        
    df=pd.concat(frames, ignore_index=True)
    log.info("Combined data shape: %s", df.shape)

    before=len(df)
    df=df.drop_duplicates()
    if len(df)<before:
        log.info("Dropped %d duplicate rows", before-len(df))

    cols_to_drop=[col for col in DROP_COLS if col in df.columns]
    df=df.drop(columns=cols_to_drop)
    log.info("Dropped columns: %s", cols_to_drop)

    df[TARGET_COL]=(df[TARGET_COL]>0).astype(int)

    df=df.dropna(subset=[TARGET_COL])

    X=df[FEATURE_COLS]
    Y=df[TARGET_COL]

    return X,Y

def build_pipeline() -> Pipeline:
    log.info("Building preprocessing pipeline")
    continuous_pipeline=Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline=Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    preprocessor=ColumnTransformer(transformers=[
        ("continuous", continuous_pipeline, CONTINUOUS_COLS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLS)
    ], remainder="drop",verbose_feature_names_out=False)

    pipeline=Pipeline([("preprocessor", preprocessor)])

    return pipeline

VALID_RANGES={
    "age": (1,120),
    "trestbps": (50,250),
    "chol": (0,600),
    "thalach":(50,250),
    "oldpeak":(0,10),
    "sex":(0,1),
    "cp":(1,4),
    "fbs":(0,1),
    "restecg":(0,2),
    "exang":(0,1)
}

def validate_input_ranges(X:pd.DataFrame)->list:
    violations=[]
    for col, (min_val, max_val) in VALID_RANGES.items():
        if col not in X.columns:
            continue
        out_of_range=(X[col]<min_val) | (X[col]>max_val)
        if out_of_range.any():
            violations.append(col)
    return violations

