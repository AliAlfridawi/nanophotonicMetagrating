# Metagrating Inverse Design CLI

A command-line interface (CLI) pipeline for the automated simulation, deep learning surrogate modeling, and inverse design of 1D silicon metagratings.

## 📖 Overview
[cite_start]Designing nanophotonic devices using conventional electromagnetic solvers like the Finite-Difference Time-Domain (FDTD) method requires significant computation time and memory consumption[cite: 104, 105]. This project bridges computational physics and artificial intelligence to bypass these bottlenecks. 

[cite_start]By generating a dataset of physical ground truths using **Meep** (an FDTD solver) and training a **PyTorch** neural network surrogate solver, this pipeline predicts the electromagnetic responses of nanophotonic structures with significantly reduced inference times[cite: 381]. Finally, it uses automatic differentiation to "inverse design" custom metagrating geometries that match a target optical spectrum.

## 🗂️ Project Structure

\`\`\`text
cli-metagrating-inverse-design/
├── data/
│   ├── raw/                    # Raw simulation outputs (.npy files)
│   └── processed/              # Cleaned and normalized PyTorch tensors (.pt files)
├── models/
│   ├── __init__.py
│   └── surrogate_mlp.py        # Multi-Layer Perceptron architecture
├── optimization/
│   ├── __init__.py
│   └── inverse_designer.py     # Automatic differentiation logic for optimization
├── simulations/
│   ├── __init__.py
│   └── data_generator.py       # Meep physics scripts using Latin Hypercube Sampling
├── main.py                     # Main CLI entry point
├── train.py                    # PyTorch training loop
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
\`\`\`

## ⚙️ Installation

Because electromagnetic solvers rely on complex C++ backends, it is highly recommended to manage your environment using `conda`.

1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/yourusername/metagrating-inverse-design.git
   cd metagrating-inverse-design
   \`\`\`

2. **Create a Conda environment and install Meep:**
   \`\`\`bash
   conda create -n inverse_design -c conda-forge pymeep
   conda activate inverse_design
   \`\`\`

3. **Install the remaining Python dependencies:**
   \`\`\`bash
   pip install torch numpy scipy tqdm
   \`\`\`

## 🚀 Usage

The entire pipeline is controlled via the `main.py` entry point. 

### Phase 1: Data Generation
Generate a dataset of random 1D silicon metagrating geometries and simulate their transmission and reflection spectra using Meep. This step utilizes Latin Hypercube Sampling to ensure uniform coverage of the parameter space.

\`\`\`bash
python main.py generate --samples 1000
\`\`\`
*Outputs: `X_inputs.npy` and `Y_outputs.npy` inside the `data/raw/` directory.*

### Phase 2: Train the Surrogate Model
Train a PyTorch Multi-Layer Perceptron (MLP) to map the geometric parameters to their corresponding optical spectra.

\`\`\`bash
python main.py train --epochs 100 --batch-size 32
\`\`\`
*Outputs: Trained model weights (e.g., `surrogate_model.pth`) saved to the `models/` directory.*

### Phase 3: Inverse Design Optimization
Input a target wavelength (or load a target spectrum array), and use automatic differentiation to backpropagate the gradients through the frozen neural network to discover the optimal physical geometry.

\`\`\`bash
python main.py optimize --target-wavelength 1550
\`\`\`
*Outputs: The optimal geometric parameters (pillar widths and gaps) printed directly to the terminal.*

## 🛠️ Built With
* **[Meep](https://meep.readthedocs.io/):** Open-source FDTD simulation software.
* **[PyTorch](https://pytorch.org/):** Deep learning tensor library.
* **[SciPy](https://scipy.org/):** Scientific computing library used for Latin Hypercube Sampling.