cd C:\Users\alfri\nanophotonicMetagrating

Write-Host "=== Step 1: Compile Check All Python Files ===" -ForegroundColor Cyan
python -m compileall main.py train.py data/contracts.py data/dataset.py optimization/inverse_designer.py simulations/verify_design.py simulations/data_generator.py 2>&1

Write-Host ""
Write-Host "=== Step 2: python main.py --help ===" -ForegroundColor Cyan
python main.py --help 2>&1

Write-Host ""
Write-Host "=== Step 3: python main.py optimize --help ===" -ForegroundColor Cyan
python main.py optimize --help 2>&1

Write-Host ""
Write-Host "=== Step 4: python main.py verify --help ===" -ForegroundColor Cyan
python main.py verify --help 2>&1
