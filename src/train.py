import logging
import os
import json
import warnings
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split, GridSearchCV, StratifiedKFold, cross_validate)
from sklearn.metrics import (balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import load_data, FEATURE_COLS, TARGET_COL, build_pipeline

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log=logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE=42
TEST_SIZE=0.2
CV_FOLDS=5

def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": balanced_accuracy_score(y_true, y_pred), 
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def make_selector():
    return SelectFromModel(RandomForestClassifier(n_estimators=100,random_state=RANDOM_STATE,),threshold="median")

def log_selected_features(pipeline:Pipeline,name:str) -> None:
    selected_mask=pipeline.named_steps["selector"].get_support()
    selected_features=[feature for feature,selected in zip(FEATURE_COLS,selected_mask) if selected]
    log.info("Selected features (%s): %s",name,selected_features)

def log_experiment(
        run_name:str,
        model_family:str,
        params:dict,
        metrics:dict,
        cv_scores:dict,
        pipeline:Pipeline
):
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_family", model_family)
        mlflow.log_params(params)

        mlflow.log_metric("balanced_accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1_score", metrics["f1_score"])

        cm_path = MODEL_DIR/f"{run_name}_confusion_matrix.json"
        with open(cm_path,"w") as f:
            json.dump(metrics["confusion_matrix"],f)
        mlflow.log_artifact(str(cm_path))

        mlflow.log_metric("cv_balanced_accuracy_mean", cv_scores["test_balanced_accuracy"].mean())
        mlflow.log_metric("cv_precision_mean", cv_scores["test_precision"].mean())
        mlflow.log_metric("cv_recall_mean", cv_scores["test_recall"].mean())
        mlflow.log_metric("cv_f1_score_mean", cv_scores["test_f1"].mean())

        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        log.info("[MLflow] %s | balanced_accuracy: %.4f | precision: %.4f | recall: %.4f | f1_score: %.4f", run_name, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"])


def train_model(
        name:str,
        model_family:str,
        model,
        X_train:pd.DataFrame,
        Y_train:pd.Series,
        X_test:pd.DataFrame,
        Y_test:pd.Series,
        extra_params:dict={},
) -> tuple[dict,dict,Pipeline]:
    log.info("Training model %s", name)
    log.info("\n"+"="*50+"\n")
    log.info("Model start: %s", name)


    preproc=build_pipeline()
    full_pipeline = Pipeline([
    ("preprocessor", preproc),
    ("selector", make_selector()),
    ("model", model)
    ])

    cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores=cross_validate(full_pipeline, X_train, Y_train, cv=cv, scoring=["balanced_accuracy", "precision", "recall", "f1"], return_train_score=False)
    log.info("CV (5-fold)| balanced_accuracy: %.4f | precision: %.4f | recall: %.4f | f1_score: %.4f", cv_scores["test_balanced_accuracy"].mean(), cv_scores["test_precision"].mean(), cv_scores["test_recall"].mean(), cv_scores["test_f1"].mean())

    full_pipeline.fit(X_train, Y_train)

    log_selected_features(full_pipeline,name)

    Y_pred=full_pipeline.predict(X_test)
    metrics=compute_metrics(Y_test, Y_pred)
    log.info("Test set | balanced_accuracy: %.4f | precision: %.4f | recall: %.4f | f1_score: %.4f", metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"]) 
    log.info("\n%s", classification_report(Y_test, Y_pred, target_names=["No Disease", "Disease"]))

    params ={"model": name,"random_state": RANDOM_STATE, **extra_params}
    log_experiment(run_name=name, model_family=model_family, params=params, metrics=metrics, cv_scores=cv_scores, pipeline=full_pipeline)
    return metrics, cv_scores, full_pipeline

def tune_random_forest(
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    X_test: pd.DataFrame,
    Y_test: pd.Series,
    ) -> tuple[dict,Pipeline]:
    log.info("Tuning Random Forest")
    log.info("\n"+"="*50+"\n")
        
    rf_pipeline = Pipeline([
    ("preprocessor", build_pipeline()),
    ("selector", make_selector()),
    ("model", RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    param_grid={
            "model__n_estimators": [100,200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5]
        }

    cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid_search=GridSearchCV(rf_pipeline, param_grid, scoring="recall", cv=cv, n_jobs=-1, verbose=0)
    grid_search.fit(X_train, Y_train)

    log.info("Best parameters: %s", grid_search.best_params_)
    log.info("Best cross-validation recall: %.4f", grid_search.best_score_)

    best_model=grid_search.best_estimator_

    log_selected_features(best_model,"RandomForest_Tuned")
    Y_pred=best_model.predict(X_test)
    metrics=compute_metrics(Y_test, Y_pred)

    log.info("Test set | balanced_accuracy: %.4f | precision: %.4f | recall: %.4f | f1_score: %.4f", metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"])
    log.info("confusion_matrix:\n%s", metrics["confusion_matrix"])

    with mlflow.start_run(run_name="RandomForest_Tuned"):
        mlflow.set_tag("model_family", "RandomForest")
        mlflow.set_tag("tuned", "True")
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("balanced_accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1_score", metrics["f1_score"])

        cm_path=MODEL_DIR/"RandomForest_Tuned_confusion_matrix.json"
        with open(cm_path,"w") as f:
            json.dump(metrics["confusion_matrix"],f)
        mlflow.log_artifact(str(cm_path))

        mlflow.sklearn.log_model(best_model, artifact_path="model")


    return metrics, best_model

def print_comparison_table(results:dict) -> None:
    rows=[]
    for name,(metrics,_) in results.items():
        rows.append({
            "Model": name,
            "Balanced Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1_score"],
            "Confusion Matrix": str(metrics["confusion_matrix"]),
        })
    df=pd.DataFrame(rows).set_index("Model")
    print(df)


    
def main():
    log.info("Starting training process")

    X, Y = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=Y)

    log.info("Data split | Train shape: %s | Test shape: %s", X_train.shape, X_test.shape)
    log.info("Train target : no disease: %d | disease: %d", (Y_train==0).sum(), (Y_train==1).sum())

    mlflow.set_experiment("Heart Disease Prediction")

    models={
        "LogisticRegression": ("LogisticRegression",LogisticRegression(max_iter=1000,random_state=RANDOM_STATE), {"max_iter":1000}),
        "SVC": ("SVC",SVC(probability=True,random_state=RANDOM_STATE), {"kernel": "rbf"}),
        "RandomForest": ("RandomForest",RandomForestClassifier(n_estimators=100,random_state=RANDOM_STATE), {"n_estimators": 100, "max_depth": None, "min_samples_split": 2,})
    }

    results={}
    pipelines={}
    for name,(family,model,extra_params) in models.items():
        metrics, cv_scores, pipeline=train_model(name, family, model, X_train, Y_train, X_test, Y_test, extra_params)
        results[name]=(metrics, cv_scores)
        pipelines[name]=pipeline

    best_rf_metrics, best_rf_model=tune_random_forest(X_train, Y_train, X_test, Y_test)
    results["RandomForest_Tuned"]=(best_rf_metrics, None)
    pipelines["RandomForest_Tuned"]=best_rf_model

    print_comparison_table(results)

    best_name=max(results,key=lambda x: results[x][0]["recall"])
    best_metrics=results[best_name][0]

    log.info("Best model: %s", best_name)
    log.info("Recall: %.4f", best_metrics["recall"])
    log.info("Precision: %.4f", best_metrics["precision"])
    log.info("F1 Score: %.4f", best_metrics["f1_score"])
    log.info("Balanced Accuracy: %.4f", best_metrics["accuracy"])

    model_path=MODEL_DIR/"best_model.pkl"
    joblib.dump(pipelines[best_name], model_path)
    log.info("Final model save -> %s",model_path)

if __name__=="__main__":
    main()
             