import logging
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import FEATURE_COLS,validate_input_ranges

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log=logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"/"best_model.pkl"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
INFERENCE_LOG=LOG_DIR/"inference.log"

MODEL_VERSION="1.0"

def load_model(model_path:Path=MODEL_DIR):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    model=joblib.load(model_path)
    log.info("Model loaded from %s", model_path)
    return model

def load_input(input_path:Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found at {input_path}")
    df=pd.read_csv(input_path)
    log.info("Input loaded from %s", input_path)
    
    df=df.drop(columns=["target"],errors="ignore")

    missing_cols=[col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input: {missing_cols}")
    
    return df[FEATURE_COLS]

def log_inference(
    input_shape:tuple,
    predictions:np.ndarray,
    probabilities:np.ndarray,
    y_true:np.ndarray=None,
):
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(INFERENCE_LOG, "a",encoding="utf-8") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model version: {MODEL_VERSION}\n")
        f.write(f"Input shape: {input_shape}\n")
        f.write(f"Predictions: {predictions.tolist()}\n")
        f.write(f"Probabilities: {probabilities.round(4).tolist()}\n")

        if y_true is not None:
            f.write(f"True labels: {y_true.tolist()}\n")
        f.write("\n")
    log.info("Inference logged to %s", INFERENCE_LOG)

def predict(
    input_path:Path,
    output_path:Path=None,
    y_true:np.ndarray=None,
) -> pd.DataFrame:
    model=load_model()
    x=load_input(input_path)
    log.info("Input shape: %s", x.shape)
    
    violations=validate_input_ranges(x)
    if violations:
        log.warning("범위 초과 특성 : %s - 예측은 계속 진행",violations)

    predictions=model.predict(x)
    probabilities=model.predict_proba(x)

    log.info("predictions | Non Disease: %d | Disease: %d", (predictions==0).sum(), (predictions==1).sum())

    log_inference(x.shape, predictions, probabilities, y_true)

    result = x.copy()
    result["prediction"] = predictions
    result["probability_no_disease"] = probabilities[:, 0].round(4)
    result["probability_disease"] = probabilities[:, 1].round(4)
    result["label"]=result["prediction"].map({0:"No Disease", 1:"Disease"})

    log.info("Result")
    for i,(pred,prob)in enumerate(zip(predictions,probabilities)):
        label="Heart Disease" if pred==1 else "No Heart Disease"
        log.info("patient %02d | %s(Disease possibilty: %1f%%)",i+1,label,prob[1]*100)

    if output_path:
        result.to_csv(output_path, index=False)
        log.info("Result saved to %s", output_path)

    return result

def main():
    parser=argparse.ArgumentParser(description="Heart Disease Prediction")
    parser.add_argument(
        "--input",type=str,
        default=str(BASE_DIR/"data"/"sample_input.csv"),
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output",type=str,default=None,
        help="Path to output CSV file"
    )
    args=parser.parse_args()
    
    input_path=Path(args.input)
    output_path=Path(args.output) if args.output else None


    result=predict(input_path, output_path)
    log.info("Total patient: %d", len(result))
    log.info("Non Disease: %d", (result["prediction"]==0).sum())
    log.info("Disease: %d", (result["prediction"]==1).sum())
    return result

if __name__=="__main__":
    result=main()





   