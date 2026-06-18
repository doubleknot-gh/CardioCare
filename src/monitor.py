import ast
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime,timedelta
from scipy.stats import ks_2samp
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import load_data,CONTINUOUS_COLS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log=logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'models'/"best_model.pkl"
LOG_DIR = BASE_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)
INFERENCE_LOG=LOG_DIR/"inference.log"
PLOT_DIR=BASE_DIR/"logs"

MODEL_VERSION="1.0"
RANDOM_STATE=42
DRIFT_THRESHOLD=0.05
TEST_SIZE=0.2
PERFORMANCE_PLOT=LOG_DIR/"performance_timeseries.png"
DRIFT_REPORT=LOG_DIR/"drift_report.csv"


def load_model(model_path:Path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    model=joblib.load(model_path)
    log.info("Model loaded from %s", model_path)
    return model

def log_inference(
    input_shape:tuple,
    predictions:np.ndarray,
    probabilities:np.ndarray,
    y_true:np.ndarray=None,
    context:str="inference",
):
    
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(INFERENCE_LOG, "a",encoding="utf-8") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model version: {MODEL_VERSION}\n")
        f.write(f"Context: {context}\n")
        f.write(f"Input shape: {input_shape}\n")
        f.write(f"Predictions: {predictions.tolist()}\n")
        f.write(f"Probabilities: {probabilities.round(4).tolist()}\n")

        if y_true is not None:
            f.write(f"True labels: {y_true.tolist()}\n")
        f.write("\n")
    log.info("Inference logged to %s", INFERENCE_LOG)


def simulate_drift(X_test:pd.DataFrame) -> pd.DataFrame:
    log.info("Simulating drift")
    X_drift=X_test.copy()
    np.random.seed(RANDOM_STATE)
    
    X_drift["chol"]=(X_drift["chol"]+30+np.random.normal(0,15,size=len(X_drift)))
    log.info("Drift Done")
    return X_drift

def detect_drift(
    X_train:pd.DataFrame,
    X_test:pd.DataFrame,
) -> pd.DataFrame:

    log.info("Detecting drift")
    rows= []
    for col in CONTINUOUS_COLS:
        if col not in X_train.columns or col not in X_test.columns:
            continue
        stat, p_value=ks_2samp(X_train[col].dropna(), X_test[col].dropna())
        drifted=p_value<DRIFT_THRESHOLD
        flag = "Drift" if drifted else "No Drift"
        log.info("%-12s|stat=%.4f|p=%.4f|%s", col, stat, p_value, flag)
        rows.append(
            {
                "feature": col,
                "stat": round(stat,4),
                "p_value": round(p_value,4),
                "drift":drifted,
            }
        )

    df_result=pd.DataFrame(rows)
    drifted_cols=df_result[df_result["drift"]]["feature"].tolist()
    log.info("Drifted columns: %s", drifted_cols if drifted_cols else "None")
    return df_result

def compare_performance(
    model,
    X_test:pd.DataFrame,
    y_test:pd.Series,
    X_drift:pd.DataFrame,
) -> dict:
    y_pred_orig=model.predict(X_test)
    y_pred_drift=model.predict(X_drift)

    if not hasattr(model,"predict_proba"):
        raise AttributeError("Loaded model does not support predict_proba().")
    proba_orig=model.predict_proba(X_test)
    proba_drift=model.predict_proba(X_drift)

    acc_orig=balanced_accuracy_score(y_test,y_pred_orig)
    acc_drift=balanced_accuracy_score(y_test,y_pred_drift)
    degradation=acc_orig-acc_drift

    log.info("Origin balanced accuracy: %.4f",acc_orig)
    log.info("Drift balanced accuracy: %.4f",acc_drift)
    log.info("Degradation: %.4f",degradation)

    log_inference(X_test.shape,y_pred_orig,proba_orig,y_test.values,context="baseline_validation")
    log_inference(X_drift.shape,y_pred_drift,proba_drift,y_test.values,context="drift_validation")

    return {"acc_original":acc_orig,"acc_drift":acc_drift,"degradation":degradation}

def parse_inference_log(log_path:Path=INFERENCE_LOG) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(columns=["timestamp","balanced_accuracy","context","model_version"])

    entries=[]
    current={}
    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line=raw_line.strip()
            if not line:
                if current:
                    entries.append(current)
                    current={}
                continue
            if ":" not in line:
                continue
            key, value=line.split(":",1)
            current[key.strip()]=value.strip()
    if current:
        entries.append(current)

    rows=[]
    for entry in entries:
        timestamp_text=entry.get("Timestamp")
        y_true_text=entry.get("True labels")
        pred_text=entry.get("Predictions")
        if not timestamp_text or not y_true_text or not pred_text:
            continue
        try:
            timestamp=datetime.strptime(timestamp_text, "%Y-%m-%d %H:%M:%S")
            y_true=np.asarray(ast.literal_eval(y_true_text))
            y_pred=np.asarray(ast.literal_eval(pred_text))
        except (ValueError, SyntaxError):
            continue
        if len(y_true) != len(y_pred) or len(y_true) == 0:
            continue
        rows.append(
            {
                "timestamp": timestamp,
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "context": entry.get("Context", "inference"),
                "model_version": entry.get("Model version", MODEL_VERSION),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["timestamp","balanced_accuracy","context","model_version"])

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def build_performance_timeseries(perf:dict) -> pd.DataFrame:
    history=parse_inference_log()
    if not history.empty:
        return history

    end_time=datetime.now().replace(second=0,microsecond=0)
    return pd.DataFrame(
        {
            "timestamp": [end_time-timedelta(days=1), end_time],
            "balanced_accuracy": [perf["acc_original"], perf["acc_drift"]],
            "context": ["baseline_validation", "drift_validation"],
            "model_version": [MODEL_VERSION, MODEL_VERSION],
        }
    )


def plot_timeseries(perf:dict,drift_result:pd.DataFrame,output_path:Path=PERFORMANCE_PLOT) -> Path:
    perf_series=build_performance_timeseries(perf)

    fig,axes=plt.subplots(2,1,figsize=(11,8),gridspec_kw={"height_ratios":[2,1]},)

    axes[0].plot(
        perf_series["timestamp"],
        perf_series["balanced_accuracy"],
        marker="o",
        linewidth=2,
        color="tab:blue",
    )
    for _, row in perf_series.iterrows():
        axes[0].annotate(
            row["context"],
            (row["timestamp"], row["balanced_accuracy"]),
            textcoords="offset points",
            xytext=(0,8),
            ha="center",
            fontsize=8,
        )
    axes[0].set_title("Balanced Accuracy Over Time")
    axes[0].set_xlabel("Timestamp")
    axes[0].set_ylabel("Balanced Accuracy")
    axes[0].set_ylim(0.0,1.05)
    axes[0].grid(True, alpha=0.3)

    if drift_result.empty:
        axes[1].text(0.5,0.5,"No drift features detected",ha="center",va="center")
        axes[1].set_title("Feature Drift p-values")
        axes[1].set_axis_off()
    else:
        colors=np.where(drift_result["drift"],"tab:red","tab:green")
        axes[1].bar(drift_result["feature"], drift_result["p_value"], color=colors)
        axes[1].axhline(DRIFT_THRESHOLD, color="black", linestyle="--", linewidth=1, label=f"threshold={DRIFT_THRESHOLD}")
        axes[1].set_title("Feature Drift p-values")
        axes[1].set_xlabel("Feature")
        axes[1].set_ylabel("p-value")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].legend()

    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("Monitoring plot saved to %s", output_path)
    return output_path


def main():
    log.info("Starting monitoring process")

    model=load_model()
    X, y=load_data()
    X_train, X_test, _, y_test=train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_drift=simulate_drift(X_test)
    drift_result=detect_drift(X_train, X_drift)
    perf=compare_performance(model, X_test, y_test, X_drift)

    drift_result.to_csv(DRIFT_REPORT, index=False)
    log.info("Drift report saved to %s", DRIFT_REPORT)

    plot_path=plot_timeseries(perf, drift_result)
    log.info("Monitoring complete | plot=%s", plot_path)
    return perf, drift_result, plot_path


if __name__=="__main__":
    main()
