# Metagrating Inverse Design CLI

## Project Purpose
This project automates inverse design for 1D silicon metagratings.  
It combines electromagnetic simulation (Meep), surrogate modeling (PyTorch), and gradient-based optimization to move from geometry sampling to optimized nanophotonic designs.

The goal is to reduce the cost of repeated FDTD simulation by training a neural surrogate that predicts spectra quickly, then using that surrogate to search for geometries that match a target optical response.

## Tech Used
- **Python**
- **Meep (pymeep)** for FDTD simulation
- **PyTorch** for surrogate neural network training and inverse optimization
- **NumPy / SciPy** for sampling and numerical workflows
- **Matplotlib** for verification plots
- **tqdm** for progress tracking

## Project Layout
```text
nanophotonicMetagrating/
├── main.py                         # Unified CLI (generate/train/optimize/verify)
├── train.py                        # Surrogate training pipeline
├── requirements.txt
├── data/
│   ├── contracts.py                # Shared constants + scaling helpers
│   ├── dataset.py                  # PyTorch dataset loader
│   ├── raw/                        # Generated X_inputs.npy, Y_outputs.npy
│   └── processed/
├── models/
│   └── surrogate_mlp.py            # MLP architecture (6 -> ... -> 22)
├── optimization/
│   └── inverse_designer.py         # Gradient-based geometry optimization
└── simulations/
    ├── data_generator.py           # Meep simulation + dataset generation
    └── verify_design.py            # FDTD vs surrogate verification + plotting
```

## Installation/Usage
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/metagrating-inverse-design.git
   cd metagrating-inverse-design
   ```

2. **Create environment and install Meep**
   ```bash
   conda create -n inverse_design -c conda-forge pymeep
   conda activate inverse_design
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the pipeline**
   ```bash
   # Phase 1: Generate simulation dataset
   python main.py generate --samples 1000

   # Phase 2: Train surrogate model
   python main.py train --epochs 100 --batch-size 32

   # Phase 3: Inverse design (example target wavelength)
   python main.py optimize --target-wavelength 1550

   # Phase 4: Verify optimized geometry with FDTD
   python main.py verify --geometry-file results/optimized_design.json --target-wavelength 1550
   ```

5. **Expected outputs**
   - `data/raw/X_inputs.npy`, `data/raw/Y_outputs.npy`
   - `models/best_surrogate.pth`
   - `results/optimized_design.json`
   - `results/validation_summary.json`
   - `results/validation_plot.png`

## Author
**Ali Alfridawi**  
