@echo off
cd /d C:\Users\alfri\nanophotonicMetagrating

echo.
echo === Step 1: Compile Check All Python Files ===
python -m compileall main.py train.py data/contracts.py data/dataset.py optimization/inverse_designer.py simulations/verify_design.py simulations/data_generator.py

echo.
echo === Step 2: python main.py --help ===
python main.py --help

echo.
echo === Step 3: python main.py optimize --help ===
python main.py optimize --help

echo.
echo === Step 4: python main.py verify --help ===
python main.py verify --help
