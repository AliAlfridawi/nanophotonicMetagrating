"""

    Fixed Parameters:
        Optical Wavelength: 1550 nm
        Grating Material: Silicon
        Silicon Refractive Index: 3.48
        Substrate Material: Silicon Dioxide
        Silicon Dioxide Refractive Index: 1.44
        Background Medium: Air
        Air Refractive Index: 1.0
        Grating Thickness: 500 nm
        Incident Light: Transverse Magnetic (TM)
        Normal Incidence: O degrees

    Variable Geometric Parameters:
        Unit Cell Period: 800-1200 nm
        Piller Widths: 50-300 nm
        Gap Widths: Spacing between pillars
        Fabrication Contraint: 50 nm 

    Output Targets:
        Transmission Spectrum: 11 discrete wavelength points evenly spaced between 1500 nm and 1600 nm
        Reflection Spectrum: Measured at those same 11 wavelenght points
        Format: Each simulation should return an array of 22 floating-point numbers

    Simulation Hyperparameters
        Grid Resolution: 20 pixels per um
        Boundary Conditions: Periodic boundary conditions on the left and right (X-axis) to simulate an infinite grating, and Perfectly Matched Layers (PML) on the top and bottom (Y-axis) to absorb outgoing waves without reflection

"""

import argparse
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm
import os

try:
    import meep as mp
except ImportError as exc:  # pragma: no cover - import guard for non-Meep environments
    mp = None
    _MEEP_IMPORT_ERROR = exc

from data.contracts import (
    MAX_FEATURE_SIZE_NM,
    MAX_PERIOD_NM,
    MIN_FEATURE_SIZE_NM,
    MIN_PERIOD_NM,
    NUM_VARIABLES,
    WAVELENGTHS_NM,
)

# Preserve legacy names used internally by this module.
WAVELENGTHS = WAVELENGTHS_NM
l_bounds = [MIN_FEATURE_SIZE_NM] * NUM_VARIABLES
u_bounds = [MAX_FEATURE_SIZE_NM] * NUM_VARIABLES


def _require_meep() -> None:
    if mp is None:
        raise RuntimeError(
            "Meep is required for electromagnetic simulation. Install pymeep in your conda environment."
        ) from _MEEP_IMPORT_ERROR


def generate_geometric_parameters(num_samples):
    valid = []
    sampler = qmc.LatinHypercube(d=NUM_VARIABLES, seed=42)

    # Oversample to account for rejections
    raw = sampler.random(n=num_samples * 5)
    scaled = qmc.scale(raw, l_bounds, u_bounds)

    for row in scaled:
        period = row.sum()
        if MIN_PERIOD_NM <= period <= MAX_PERIOD_NM:
            valid.append(row)
        if len(valid) == num_samples:
            break

    if len(valid) < num_samples:
        raise RuntimeError(f"Only {len(valid)} valid samples found. Increase oversampling factor.")

    return np.array(valid)

def run_electromagnetic_simulation(geometry_params):
    """
    Runs a two-pass MEEP simulation (normalization + structure) for a
    3-pillar silicon grating on a SiO2 substrate.

    Args:
        geometry_params: np.array of shape (6,) — [w1, w2, w3, g1, g2, g3] in nm

    Returns:
        np.array of shape (22,) — [T(11 freqs), R(11 freqs)], both normalized to [0, 1]
    """

    _require_meep()

    # ------------------------------------------------------------------ #
    # 1. UNPACK & CONVERT PARAMETERS  (nm → µm)                          #
    # ------------------------------------------------------------------ #
    w1, w2, w3, g1, g2, g3 = geometry_params / 1000.0
    period = w1 + w2 + w3 + g1 + g2 + g3

    # ------------------------------------------------------------------ #
    # 2. CELL & FIXED GEOMETRY CONSTANTS                                  #
    # ------------------------------------------------------------------ #
    h      = 0.500   # Grating thickness:        500 nm
    dpml   = 1.000   # PML thickness:              1 µm
    air_gap = 2.000  # Air above/below structure:  2 µm

    sx = period
    sy = h + 2 * dpml + 2 * air_gap   # total cell height
    cell = mp.Vector3(sx, sy, 0)

    # Vertical centre of the grating slab sits at y = 0
    # Everything below y = 0 down to the bottom PML is substrate
    substrate_height = air_gap + dpml          # region below grating midplane
    substrate_cy     = -(h / 2 + substrate_height / 2)

    # ------------------------------------------------------------------ #
    # 3. MATERIALS                                                        #
    # ------------------------------------------------------------------ #
    Si   = mp.Medium(index=3.48)
    SiO2 = mp.Medium(index=1.44)

    # ------------------------------------------------------------------ #
    # 4. GEOMETRY                                                         #
    # ------------------------------------------------------------------ #
    # Pillar centres — packed left-to-right inside the unit cell
    x1 = -period / 2 + w1 / 2
    x2 =  x1 + w1 / 2 + g1 + w2 / 2
    x3 =  x2 + w2 / 2 + g2 + w3 / 2

    geometry = [
        # SiO2 substrate fills everything below the grating layer
        mp.Block(
            size=mp.Vector3(sx, substrate_height, mp.inf),
            center=mp.Vector3(0, substrate_cy, 0),
            material=SiO2
        ),
        # Silicon pillars
        mp.Block(mp.Vector3(w1, h, mp.inf), center=mp.Vector3(x1, 0, 0), material=Si),
        mp.Block(mp.Vector3(w2, h, mp.inf), center=mp.Vector3(x2, 0, 0), material=Si),
        mp.Block(mp.Vector3(w3, h, mp.inf), center=mp.Vector3(x3, 0, 0), material=Si),
    ]

    # ------------------------------------------------------------------ #
    # 5. SOURCE  (broadband Gaussian, TM = Ez)                           #
    # ------------------------------------------------------------------ #
    # Source sits in the air gap, just below the upper PML
    src_y    = sy / 2 - dpml - 0.5

    fcen_src = 1.0 / 1.55   # centre frequency (1/µm), ~1550 nm
    df_src   = 0.30          # wide enough to cover 1500–1600 nm

    sources = [
        mp.Source(
            mp.GaussianSource(frequency=fcen_src, fwidth=df_src),
            component=mp.Ez,                          # TM polarisation
            center=mp.Vector3(0, src_y, 0),
            size=mp.Vector3(sx, 0, 0)                 # plane wave across full period
        )
    ]

    # ------------------------------------------------------------------ #
    # 6. SHARED SIMULATION SETTINGS                                       #
    # ------------------------------------------------------------------ #
    resolution  = 20                         # pixels / µm
    pml_layers  = [mp.PML(dpml, direction=mp.Y)]
    k_point     = mp.Vector3(0, 0, 0)        # normal incidence + periodic BCs on X

    # Flux monitor frequency axis — 11 points spanning 1500–1600 nm
    fmin    = 1.0 / 1.600   # lowest  freq  (longest λ)
    fmax    = 1.0 / 1.500   # highest freq  (shortest λ)
    fcen_mon = (fmin + fmax) / 2
    df_mon   = fmax - fmin
    nfreq   = 11

    # Monitor positions
    # Reflection monitor: between source and grating (just below source)
    refl_y  = src_y - 0.1
    # Transmission monitor: in substrate, just above bottom PML
    tran_y  = -sy / 2 + dpml + 0.5

    decay_pt = mp.Vector3(0, tran_y, 0)   # field-decay check point

    refl_reg = mp.FluxRegion(center=mp.Vector3(0, refl_y, 0), size=mp.Vector3(sx, 0, 0))
    tran_reg = mp.FluxRegion(center=mp.Vector3(0, tran_y, 0), size=mp.Vector3(sx, 0, 0))

    # ------------------------------------------------------------------ #
    # 7. PASS 1 — NORMALISATION RUN  (empty cell, no geometry)           #
    # Records the raw incident flux so we can normalise T and R to [0,1] #
    # ------------------------------------------------------------------ #
    sim = mp.Simulation(
        cell_size=cell,
        geometry=[],                  # no structure
        sources=sources,
        boundary_layers=pml_layers,
        k_point=k_point,
        resolution=resolution
    )

    norm_refl_flux = sim.add_flux(fcen_mon, df_mon, nfreq, refl_reg)
    norm_tran_flux = sim.add_flux(fcen_mon, df_mon, nfreq, tran_reg)

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ez, decay_pt, 1e-3
        )
    )

    # Save incident flux values and the DFT fields for later subtraction
    incident_flux  = np.array(mp.get_fluxes(norm_tran_flux))   # for normalising T
    incident_data  = sim.get_flux_data(norm_refl_flux)          # for subtracting from R
    sim.reset_meep()

    # ------------------------------------------------------------------ #
    # 8. PASS 2 — STRUCTURE RUN                                           #
    # ------------------------------------------------------------------ #
    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=sources,
        boundary_layers=pml_layers,
        k_point=k_point,
        resolution=resolution
    )

    tran_flux = sim.add_flux(fcen_mon, df_mon, nfreq, tran_reg)
    refl_flux = sim.add_flux(fcen_mon, df_mon, nfreq, refl_reg)

    # Subtract the saved incident fields so the reflection monitor measures
    # only the back-scattered light, not the forward-travelling wave
    sim.load_minus_flux_data(refl_flux, incident_data)

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ez, decay_pt, 1e-3
        )
    )

    # ------------------------------------------------------------------ #
    # 9. EXTRACT & NORMALISE                                              #
    # ------------------------------------------------------------------ #
    # Divide by incident flux → T and R each live in [0, 1]
    T =  np.array(mp.get_fluxes(tran_flux)) / incident_flux
    R = -np.array(mp.get_fluxes(refl_flux)) / incident_flux   # minus: reflected flux travels −y

    sim.reset_meep()

    # Sanity check — energy conservation (T + R ≤ 1, allowing small numerical error)
    if np.any(T + R > 1.05):
        print(f"[WARNING] T + R > 1 detected. Max value: {(T + R).max():.4f}. "
              f"Check geometry_params: {geometry_params}")

    return np.concatenate((T, R))   # shape (22,)

def generate_dataset(
    num_samples: int,
    output_dir: str = "data/raw",
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    _require_meep()

    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")

    x_data = generate_geometric_parameters(num_samples)
    y_data = np.zeros((num_samples, len(WAVELENGTHS) * 2), dtype=np.float32)

    iterator = range(num_samples)
    if show_progress:
        iterator = tqdm(iterator, desc="Simulating")

    for i in iterator:
        y_data[i] = run_electromagnetic_simulation(x_data[i])

    os.makedirs(output_dir, exist_ok=True)
    x_path = os.path.join(output_dir, "X_inputs.npy")
    y_path = os.path.join(output_dir, "Y_outputs.npy")
    np.save(x_path, x_data)
    np.save(y_path, y_data)
    return x_data, y_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Meep simulation data for metagrating design.")
    parser.add_argument("--samples", type=int, default=100, help="Number of valid geometries to simulate.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory where X_inputs.npy and Y_outputs.npy will be saved.",
    )
    args = parser.parse_args()

    print("Running simulations...")
    generate_dataset(num_samples=args.samples, output_dir=args.output_dir, show_progress=True)
    print(f"\nDataset successfully generated and saved to '{args.output_dir}'.")

if __name__ == "__main__":
    main()
