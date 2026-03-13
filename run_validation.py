#!/usr/bin/env python3
import subprocess
import sys
import os

os.chdir(r'C:\Users\alfri\nanophotonicMetagrating')

# Step 1: Compile check
print("=== Step 1: Compile Check All Python Files ===")
files = [
    'main.py',
    'train.py',
    'data/contracts.py',
    'data/dataset.py',
    'optimization/inverse_designer.py',
    'simulations/verify_design.py',
    'simulations/data_generator.py'
]
result = subprocess.run([sys.executable, '-m', 'compileall'] + files, capture_output=False)
print(f"Compileall exit code: {result.returncode}\n")

# Step 2: main.py --help
print("=== Step 2: python main.py --help ===")
result = subprocess.run([sys.executable, 'main.py', '--help'], capture_output=False)
print(f"Exit code: {result.returncode}\n")

# Step 3: main.py optimize --help
print("=== Step 3: python main.py optimize --help ===")
result = subprocess.run([sys.executable, 'main.py', 'optimize', '--help'], capture_output=False)
print(f"Exit code: {result.returncode}\n")

# Step 4: main.py verify --help
print("=== Step 4: python main.py verify --help ===")
result = subprocess.run([sys.executable, 'main.py', 'verify', '--help'], capture_output=False)
print(f"Exit code: {result.returncode}\n")
