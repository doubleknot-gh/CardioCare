# 🫀 CardioCare — 심장병 예측 종단간 ML 시스템

> **UCI Heart Disease Dataset**을 활용한 임상 의사결정 보조 시스템
> 본 시스템은 심장 전문의의 판단을 **보조(inform)** 하는 도구이며, 절대 단독으로 진단을 **결정(decide)** 하지 않습니다.

---

## 📁 프로젝트 구조

```
CardioCare/
├── data/
│   ├── processed.cleveland.data
│   ├── processed.hungarian.data
│   ├── processed.switzerland.data
│   ├── processed.va.data
│   └── sample_input.csv
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   └── 01_eda_preprocessing.executed.ipynb
├── src/
│   ├── preprocessing.py
│   ├── mlflow_config.py
│   ├── train.py
│   ├── inference.py
│   └── monitor.py
├── tests/
│   └── test_pipeline.py
├── models/
│   └── best_model.pkl
├── mlruns/                  # legacy file-store runs (optional migration source)
├── mlartifacts/
├── mlflow.db
├── logs/
├── start_mlflow_ui.ps1
├── Dockerfile
├── requirements.txt
├── .github/workflows/ci.yml
├── report.docx
├── report.pdf
└── README.md
```

---

## ⚙️ 환경 요구사항

| 항목 | 버전 |
|------|------|
| Python | 3.11 이상 |
| Docker | 24.0 이상 |

---

## 🚀 전체 재현 절차

### 0단계 — 저장소 클론 및 환경 설정

```bash
git clone https://github.com/<your-username>/CardioCare.git
cd CardioCare
pip install -r requirements.txt
```

### 1단계 — 모델 학습

```bash
python src/train.py
```

완료되면:
- `models/best_model.pkl` 자동 생성
- MLflow 실험 기록 → `mlflow.db`
- MLflow 아티팩트 → `mlartifacts/`
- 최종 선택 모델은 MLflow run tag `selected_model=true`로 표시

MLflow UI 확인:
```powershell
.\start_mlflow_ui.ps1
# → http://127.0.0.1:5000
```

직접 실행:
```powershell
mlflow ui --backend-store-uri "sqlite:///mlflow.db" --port 5000
```

UI 확인 포인트:
- `Experiments` → `Heart Disease Prediction`
- `Training Runs`에 `LogisticRegression`, `SVC`, `RandomForest`, `RandomForest_Tuned` 표시
- 최종 선택 모델 run의 `Tags`에 `selected_model=true`, `model_role=final_candidate` 표시

기존 `mlruns/` file-store 기록이 있다면 1회 마이그레이션:
```powershell
mlflow migrate-filestore --source .\mlruns --target "sqlite:///mlflow.db" --progress
```

### 2단계 — 단위 테스트

```bash
python -m unittest discover tests/ -v
```

예상 출력:
```
test_no_missing_values_after_preprocessing ... ok
test_pipeline_deterministic_output ... ok
test_prediction_probability_range ... ok
test_prediction_probability_shape ... ok
test_prediction_probability_sum ... ok
test_prediction_shape ... ok
test_preprocessing_output_shape ... ok
test_validate_input_ranges ... ok

Ran 8 tests in 0.5s OK
```

### 3단계 — 로컬 추론

```bash
python src/inference.py --input data/sample_input.csv --output logs/predictions.csv
```

완료되면:
- `logs/inference.log` — 추론 로그
- `logs/predictions.csv` — 예측 결과 CSV

### 4단계 — Docker 빌드 및 추론

```bash
docker build -t cardiocare:1.0 .

# 실행
docker run --rm -v $(pwd)/data:/app/data cardiocare:1.0
```

Windows PowerShell:
```powershell
docker run --rm -v ${PWD}/data:/app/data cardiocare:1.0
```

### 5단계 — 모니터링 및 드리프트 탐지

```bash
python src/monitor.py
```

완료되면:
- `logs/inference.log` — 추론 로그
- `logs/drift_report.csv` — KS 검정 결과
- `logs/performance_timeseries.png` — 성능 시계열 그래프

---

## 📊 모델 성능 (실제 결과)

| 모델 | Balanced Acc | Precision | Recall | F1 |
|------|-------------|-----------|--------|----|
| Logistic Regression | 0.7972 | 0.8000 | 0.8627 | 0.8302 |
| **SVC (최종 선택)** | **0.7960** | **0.7946** | **0.8725** | **0.8318** |
| Random Forest | 0.7849 | 0.8058 | 0.8137 | 0.8098 |
| RandomForest Tuned | 0.7776 | 0.7925 | 0.8235 | 0.8077 |

**최종 모델: SVC** — Recall 0.8725로 최고 (FN 최소화)

---

## 🔑 데이터 누수 방지 원칙

```python
# ❌ 잘못된 예
scaler.fit(X)
X_scaled = scaler.transform(X)
X_train, X_test = train_test_split(X_scaled)

# ✅ 올바른 예
X_train, X_test = train_test_split(X)
scaler.fit(X_train)        # train에만 fit
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)
```

---

## 🔒 재현성

모든 랜덤 시드 고정:
```python
RANDOM_STATE = 42
```

---

## 🩺 윤리적 고려사항

> **CardioCare는 심장 전문의의 의사결정을 보조하는 도구입니다.**
> 절대 단독으로 진단을 결정하지 않습니다.

---

## 📋 채점자 재현 체크포인트

```bash
# 1. 학습
python src/train.py

# 2. MLflow 확인
.\start_mlflow_ui.ps1  # → http://127.0.0.1:5000

# 3. 테스트
python -m unittest discover tests/ -v

# 4. 로컬 추론
python src/inference.py --input data/sample_input.csv --output logs/predictions.csv

# 5. Docker
docker build -t cardiocare:1.0 .
docker run --rm -v $(pwd)/data:/app/data cardiocare:1.0

# 6. 드리프트
python src/monitor.py
```

---

## 🤖 AI 도구 사용 공개

프로젝트에서 Claude를 코드 작성 및 디버깅 보조, Word 템플릿 디자인, README 작성 목적으로 사용했다. 모든 설계 결정, 모델 선택 근거, 분석 해석, 보고서 작성은 나 자신이 수행했음을 알림.

---
