# Nanophotonic Metagrating Inverse Design

> A simulation-to-optimization workflow for 1D silicon metagratings using Meep, PyTorch, and gradient-based inverse design.

This repository packages the full metagrating workflow into a single CLI:

| Stage | Purpose |
| --- | --- |
| `generate` | Sample physically valid geometries and simulate spectra with Meep |
| `train` | Train a surrogate neural network to predict optical response |
| `optimize` | Search for a geometry that matches a target spectrum |
| `verify` | Re-run the final design with FDTD and compare prediction vs physics |

---

## Project Purpose

This project automates inverse design for **1D silicon metagratings**.
It combines electromagnetic simulation, surrogate modeling, and differentiable optimization to move from raw geometry sampling to validated nanophotonic designs.

The main objective is to reduce the cost of repeated full-wave simulation by training a neural surrogate that predicts spectra quickly, then using that surrogate to search for geometries that match a desired optical response.

In practical terms, the repository is designed to help you:

- generate a dataset of physically valid metagrating geometries
- train a PyTorch model to approximate Meep simulation results
- optimize geometry parameters against a target response
- verify the optimized design against FDTD ground truth

---

## Tech Used

| Technology | Why it is used |
| --- | --- |
| **Python** | Main language for the CLI, data flow, and orchestration |
| **Meep (pymeep)** | FDTD simulation engine for transmission/reflection spectra |
| **PyTorch** | Surrogate model training and inverse optimization |
| **NumPy** | Array operations, dataset storage, and numerical preprocessing |
| **SciPy** | Latin Hypercube Sampling and scientific utilities |
| **Matplotlib** | Plotting spectra and saving validation figures |
| **tqdm** | Progress bars for long-running simulation loops |

---

## Project Layout

```text
nanophotonicMetagrating/
|-- main.py                      # Unified CLI entrypoint
|-- train.py                     # Surrogate training pipeline
|-- run_validation.py            # Validation helper script
|-- requirements.txt             # Runtime Python dependencies
|-- validation_script.ps1        # PowerShell validation helper
|-- validation_script.bat        # Windows batch validation helper
|-- data/
|   |-- __init__.py
|   |-- contracts.py             # Shared constants and scaling helpers
|   |-- dataset.py               # PyTorch dataset loader
|   |-- raw/                     # Generated X_inputs.npy and Y_outputs.npy
|   `-- processed/
|-- models/
|   |-- __init__.py
|   `-- surrogate_mlp.py         # MLP surrogate model
|-- optimization/
|   |-- __init__.py
|   `-- inverse_designer.py      # Gradient-based geometry optimization
|-- simulations/
|   |-- __init__.py
|   |-- data_generator.py        # Meep simulation and dataset generation
|   `-- verify_design.py         # FDTD vs surrogate verification
`-- docs/
    |-- phaseOne.md
    |-- phaseTwo.md
    |-- phaseThree.md
    `-- phaseFour.md
```

---

## Installation/Usage

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/nanophotonicMetagrating.git
cd nanophotonicMetagrating
```

### 2. Create the environment

Install Meep in a Conda environment first:

```bash
conda create -n inverse_design -c conda-forge pymeep
conda activate inverse_design
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
# Phase 1: Generate simulation dataset
python main.py generate --samples 1000

# Phase 2: Train surrogate model
python main.py train --epochs 100 --batch-size 32

# Phase 3: Inverse design for a target wavelength
python main.py optimize --target-wavelength 1550

# Phase 4: Verify optimized geometry with FDTD
python main.py verify --geometry-file results/optimized_design.json --target-wavelength 1550
```

### 5. Expected outputs

After a successful run, you should expect artifacts such as:

- `data/raw/X_inputs.npy`
- `data/raw/Y_outputs.npy`
- `models/best_surrogate.pth`
- `results/optimized_design.json`
- `results/validation_summary.json`
- `results/validation_plot.png`

### 6. Validation helpers

You can also run the repository validation helpers:

```bash
python run_validation.py
```

On Windows:

```powershell
./validation_script.ps1
```

or:

```bat
validation_script.bat
```

---

## Author

**Ali Alfridawi**
