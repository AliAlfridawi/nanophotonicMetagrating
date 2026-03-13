from __future__ import annotations

import argparse
import json
import os
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn as nn

from data.contracts import (
    NUM_VARIABLES,
    NUM_WAVELENGTHS,
    SPECTRUM_DIM,
    WAVELENGTHS_NM,
    build_bandpass_target_spectrum,
    is_valid_geometry_nm,
    scale_geometry_nm_to_unit,
)
from models.surrogate_mlp import SurrogateMLP
from simulations.data_generator import run_electromagnetic_simulation

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_geometry_from_file(geometry_file: str) -> np.ndarray:
    if not os.path.exists(geometry_file):
        raise FileNotFoundError(f"Geometry file not found: {geometry_file}")

    if geometry_file.endswith(".json"):
        with open(geometry_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "geometry_nm" not in payload:
            raise ValueError("JSON file must contain 'geometry_nm'.")
        geometry = np.asarray(payload["geometry_nm"], dtype=np.float32)
    elif geometry_file.endswith(".npy"):
        geometry = np.load(geometry_file).astype(np.float32)
    else:
        raise ValueError("geometry_file must be .json or .npy")

    if geometry.shape != (NUM_VARIABLES,):
        raise ValueError(f"Expected geometry shape ({NUM_VARIABLES},), got {geometry.shape}")
    return geometry


def _parse_geometry(geometry_csv: str | None, geometry_file: str | None) -> np.ndarray:
    if geometry_csv is None and geometry_file is None:
        raise ValueError("Provide either --geometry or --geometry-file.")
    if geometry_csv is not None and geometry_file is not None:
        raise ValueError("Use either --geometry or --geometry-file, not both.")

    if geometry_file:
        geometry = _load_geometry_from_file(geometry_file)
    else:
        parts = [p.strip() for p in geometry_csv.split(",")]
        if len(parts) != NUM_VARIABLES:
            raise ValueError(f"--geometry requires {NUM_VARIABLES} comma-separated values.")
        geometry = np.asarray([float(p) for p in parts], dtype=np.float32)

    if not is_valid_geometry_nm(geometry):
        raise ValueError(
            "Geometry violates physical constraints (feature bounds or period). "
            f"Received: {geometry}"
        )
    return geometry


def _load_surrogate_prediction(geometry_nm: np.ndarray, model_path: str, device: torch.device) -> np.ndarray:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    model = SurrogateMLP().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        x_unit = scale_geometry_nm_to_unit(geometry_nm).astype(np.float32)
        x_tensor = torch.from_numpy(x_unit).unsqueeze(0).to(device)
        y_pred = model(x_tensor).squeeze(0).cpu().numpy()
    return y_pred


def verify_design(
    geometry_nm: np.ndarray,
    model_path: str,
    output_plot: str = "results/validation_plot.png",
    output_json: str = "results/validation_summary.json",
    target_wavelength_nm: float | None = None,
    target_spectrum_path: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    if target_wavelength_nm is not None and target_spectrum_path is not None:
        raise ValueError("Use either target_wavelength_nm or target_spectrum_path, not both.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if target_spectrum_path and not os.path.exists(target_spectrum_path):
        raise FileNotFoundError(f"Target spectrum file not found: {target_spectrum_path}")

    run_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fdtd_spectrum = run_electromagnetic_simulation(geometry_nm).astype(np.float32)
    surrogate_spectrum = _load_surrogate_prediction(geometry_nm, model_path, run_device).astype(np.float32)

    mse_pred_vs_fdtd = float(nn.MSELoss()(torch.from_numpy(surrogate_spectrum), torch.from_numpy(fdtd_spectrum)).item())

    target = None
    if target_spectrum_path:
        target = np.load(target_spectrum_path).astype(np.float32)
    elif target_wavelength_nm is not None:
        target = build_bandpass_target_spectrum(target_wavelength_nm)

    if target is not None and target.shape != (SPECTRUM_DIM,):
        raise ValueError(f"Target spectrum must have shape ({SPECTRUM_DIM},), got {target.shape}")

    mse_target_vs_pred = None
    mse_target_vs_fdtd = None
    if target is not None:
        target_tensor = torch.from_numpy(target)
        mse_target_vs_pred = float(nn.MSELoss()(target_tensor, torch.from_numpy(surrogate_spectrum)).item())
        mse_target_vs_fdtd = float(nn.MSELoss()(target_tensor, torch.from_numpy(fdtd_spectrum)).item())

    os.makedirs(os.path.dirname(output_plot) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)

    target_t = target[:NUM_WAVELENGTHS] if target is not None else None
    target_r = target[NUM_WAVELENGTHS:] if target is not None else None
    pred_t = surrogate_spectrum[:NUM_WAVELENGTHS]
    pred_r = surrogate_spectrum[NUM_WAVELENGTHS:]
    fdtd_t = fdtd_spectrum[:NUM_WAVELENGTHS]
    fdtd_r = fdtd_spectrum[NUM_WAVELENGTHS:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    axes[0].plot(WAVELENGTHS_NM, pred_t, label="Surrogate Predicted T", linewidth=2)
    axes[0].plot(WAVELENGTHS_NM, fdtd_t, label="FDTD Actual T", linewidth=2, linestyle="--")
    if target_t is not None:
        axes[0].plot(WAVELENGTHS_NM, target_t, label="Target T", linewidth=2, linestyle=":")
    axes[0].set_title("Transmission")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Efficiency")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(WAVELENGTHS_NM, pred_r, label="Surrogate Predicted R", linewidth=2)
    axes[1].plot(WAVELENGTHS_NM, fdtd_r, label="FDTD Actual R", linewidth=2, linestyle="--")
    if target_r is not None:
        axes[1].plot(WAVELENGTHS_NM, target_r, label="Target R", linewidth=2, linestyle=":")
    axes[1].set_title("Reflection")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Inverse Design Verification")
    fig.tight_layout()
    fig.savefig(output_plot, dpi=200)
    plt.close(fig)

    summary = {
        "geometry_nm": geometry_nm.astype(float).tolist(),
        "mse_pred_vs_fdtd": mse_pred_vs_fdtd,
        "mse_target_vs_pred": mse_target_vs_pred,
        "mse_target_vs_fdtd": mse_target_vs_fdtd,
        "output_plot": output_plot,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify inverse-designed geometry using FDTD simulation.")
    parser.add_argument(
        "--geometry",
        type=str,
        default=None,
        help="Comma-separated geometry in nm: w1,w2,w3,g1,g2,g3",
    )
    parser.add_argument(
        "--geometry-file",
        type=str,
        default=None,
        help="Path to .json or .npy file containing geometry.",
    )
    parser.add_argument("--model-path", type=str, default="models/best_surrogate.pth")
    parser.add_argument("--target-wavelength", type=float, default=None)
    parser.add_argument("--target-spectrum-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-plot", type=str, default="results/validation_plot.png")
    parser.add_argument("--output-json", type=str, default="results/validation_summary.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    geometry_nm = _parse_geometry(args.geometry, args.geometry_file)
    summary = verify_design(
        geometry_nm=geometry_nm,
        model_path=args.model_path,
        output_plot=args.output_plot,
        output_json=args.output_json,
        target_wavelength_nm=args.target_wavelength,
        target_spectrum_path=args.target_spectrum_path,
        device=args.device,
    )
    print("Verification complete.")
    print(f"MSE (Pred vs FDTD): {summary['mse_pred_vs_fdtd']:.6f}")
    if summary["mse_target_vs_pred"] is not None:
        print(f"MSE (Target vs Pred): {summary['mse_target_vs_pred']:.6f}")
        print(f"MSE (Target vs FDTD): {summary['mse_target_vs_fdtd']:.6f}")
    print(f"Saved plot -> {summary['output_plot']}")
    print(f"Saved summary -> {args.output_json}")


if __name__ == "__main__":
    main()

