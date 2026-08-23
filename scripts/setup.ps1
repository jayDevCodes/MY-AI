$ErrorActionPreference = "Stop"

$Python = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$Venv = if ($env:VENV_DIR) { $env:VENV_DIR } else { ".venv" }

& $Python -m venv $Venv
& "$Venv\Scripts\python.exe" -m pip install --upgrade pip
& "$Venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
& "$Venv\Scripts\python.exe" -m pip install -e .

Write-Host "MY-AI environment ready. Activate with: $Venv\Scripts\Activate.ps1"
Write-Host "Run tests with: pytest"
