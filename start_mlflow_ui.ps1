param(
    [int]$Port = 5000
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$DatabasePath = Join-Path $ProjectRoot "mlflow.db"
$BackendStoreUri = "sqlite:///$($DatabasePath -replace '\\', '/')"

if (-not (Test-Path $DatabasePath)) {
    Write-Warning "mlflow.db not found. Run 'python src/train.py' or migrate existing mlruns first."
}

Write-Host "Starting MLflow UI with backend: $BackendStoreUri"
mlflow ui --backend-store-uri $BackendStoreUri --port $Port
