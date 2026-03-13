from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from data.contracts import (
    MAX_PERIOD_NM,
    MIN_PERIOD_NM,
    NUM_VARIABLES,
    SPECTRUM_DIM,
    WAVELENGTHS_NM,
    build_bandpass_target_spectrum,
    unscale_geometry_unit_to_nm,
)
from models.surrogate_mlp import SurrogateMLP


def load_frozen_model(model_path: str, device: torch.device) -> SurrogateMLP:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    model = SurrogateMLP().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_target_tensor(
    device: torch.device,
    target_wavelength_nm: float | None = None,
    target_spectrum_path: str | None = None,
) -> torch.Tensor:
    if target_spectrum_path is None and target_wavelength_nm is None:
        raise ValueError("Provide either target_wavelength_nm or target_spectrum_path.")
    if target_spectrum_path is not None and target_wavelength_nm is not None:
        raise ValueError("Use either target_wavelength_nm or target_spectrum_path, not both.")

    if target_spectrum_path is not None:
        if not os.path.exists(target_spectrum_path):
            raise FileNotFoundError(f"Target spectrum file not found: {target_spectrum_path}")
        target = np.load(target_spectrum_path).astype(np.float32)
        if target.shape != (SPECTRUM_DIM,):
            raise ValueError(f"Target spectrum must have shape ({SPECTRUM_DIM},), got {target.shape}")
    else:
        target = build_bandpass_target_spectrum(float(target_wavelength_nm))

    return torch.from_numpy(target).to(device)


def inverse_design(
    model_path: str,
    output_json: str | None = None,
    target_wavelength_nm: float | None = None,
    target_spectrum_path: str | None = None,
    steps: int = 500,
    learning_rate: float = 0.03,
    period_penalty_weight: float = 5.0,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be > 0.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")

    torch.manual_seed(seed)
    run_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_frozen_model(model_path=model_path, device=run_device)
    target = build_target_tensor(
        device=run_device,
        target_wavelength_nm=target_wavelength_nm,
        target_spectrum_path=target_spectrum_path,
    ).unsqueeze(0)

    geometry_unit = torch.rand((1, NUM_VARIABLES), device=run_device, requires_grad=True)
    optimizer = torch.optim.Adam([geometry_unit], lr=learning_rate)
    criterion = nn.MSELoss()

    best_total_loss = float("inf")
    best_geometry_unit = None
    best_prediction = None

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        prediction = model(geometry_unit)
        spectrum_loss = criterion(prediction, target)

        geometry_nm = unscale_geometry_unit_to_nm(geometry_unit)
        period_nm = torch.sum(geometry_nm, dim=1)
        period_low_violation = torch.relu(torch.tensor(MIN_PERIOD_NM, device=run_device) - period_nm)
        period_high_violation = torch.relu(period_nm - torch.tensor(MAX_PERIOD_NM, device=run_device))
        period_penalty = torch.mean((period_low_violation + period_high_violation) ** 2)

        total_loss = spectrum_loss + period_penalty_weight * period_penalty
        current_loss = float(total_loss.item())
        if current_loss < best_total_loss:
            best_total_loss = current_loss
            best_geometry_unit = geometry_unit.detach().clone()
            best_prediction = prediction.detach().clone()

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            geometry_unit.clamp_(0.0, 1.0)

        if step % 50 == 0 or step == 1 or step == steps:
            print(
                f"Step {step:04d}/{steps} | total={current_loss:.6f} "
                f"| spectrum={spectrum_loss.item():.6f} | period_penalty={period_penalty.item():.6f}"
            )

    if best_geometry_unit is None or best_prediction is None:
        raise RuntimeError("Inverse design failed to produce an optimized geometry.")

    best_geometry_nm = unscale_geometry_unit_to_nm(best_geometry_unit).squeeze(0).cpu().numpy()
    best_prediction_np = best_prediction.squeeze(0).cpu().numpy()
    target_np = target.squeeze(0).cpu().numpy()
    period_nm = float(np.sum(best_geometry_nm))

    result = {
        "best_total_loss": best_total_loss,
        "period_nm": period_nm,
        "geometry_nm": best_geometry_nm.tolist(),
        "predicted_spectrum": best_prediction_np.tolist(),
        "target_spectrum": target_np.tolist(),
        "wavelengths_nm": WAVELENGTHS_NM.astype(float).tolist(),
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved optimization result -> {output_json}")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inverse-design metagrating geometry using a trained surrogate.")
    parser.add_argument("--model-path", default="models/best_surrogate.pth")
    parser.add_argument("--target-wavelength", type=float, default=None, help="Bandpass target center wavelength in nm.")
    parser.add_argument(
        "--target-spectrum-path",
        type=str,
        default=None,
        help=f"Optional .npy target spectrum path with shape ({SPECTRUM_DIM},).",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--period-penalty-weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=str, default="results/optimized_design.json")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = inverse_design(
        model_path=args.model_path,
        output_json=args.output_json,
        target_wavelength_nm=args.target_wavelength,
        target_spectrum_path=args.target_spectrum_path,
        steps=args.steps,
        learning_rate=args.learning_rate,
        period_penalty_weight=args.period_penalty_weight,
        seed=args.seed,
        device=args.device,
    )
    print(f"Final loss: {result['best_total_loss']:.6f}")
    print(f"Period (nm): {result['period_nm']:.2f}")
    print(f"Optimized geometry [w1,w2,w3,g1,g2,g3] (nm): {np.array(result['geometry_nm'])}")


if __name__ == "__main__":
    main()

