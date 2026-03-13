from __future__ import annotations

import argparse
import numpy as np


def _cmd_generate(args: argparse.Namespace) -> None:
    from simulations.data_generator import generate_dataset

    generate_dataset(num_samples=args.samples, output_dir=args.output_dir, show_progress=True)
    print(f"Generated dataset with {args.samples} samples at '{args.output_dir}'.")


def _cmd_train(args: argparse.Namespace) -> None:
    from train import train_surrogate

    metrics = train_surrogate(
        x_path=args.x_path,
        y_path=args.y_path,
        output_model=args.output_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        seed=args.seed,
        device=args.device,
    )
    print(f"Training complete. Best validation MSE: {metrics['best_val_loss']:.6f}")


def _cmd_optimize(args: argparse.Namespace) -> None:
    from optimization.inverse_designer import inverse_design

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
    geometry_nm = np.array(result["geometry_nm"])
    print(f"Inverse design complete. Final loss: {result['best_total_loss']:.6f}")
    print(f"Optimized geometry [w1,w2,w3,g1,g2,g3] (nm): {geometry_nm}")
    print(f"Period (nm): {result['period_nm']:.2f}")


def _cmd_verify(args: argparse.Namespace) -> None:
    from simulations.verify_design import _parse_geometry, verify_design

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
    print(f"Saved plot -> {summary['output_plot']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Metagrating inverse design pipeline CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate FDTD dataset with Meep.")
    generate_parser.add_argument("--samples", type=int, default=100)
    generate_parser.add_argument("--output-dir", type=str, default="data/raw")
    generate_parser.set_defaults(handler=_cmd_generate)

    train_parser = subparsers.add_parser("train", help="Train surrogate model from generated dataset.")
    train_parser.add_argument("--x-path", default="data/raw/X_inputs.npy")
    train_parser.add_argument("--y-path", default="data/raw/Y_outputs.npy")
    train_parser.add_argument("--output-model", default="models/best_surrogate.pth")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--val-split", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", type=str, default=None)
    train_parser.set_defaults(handler=_cmd_train)

    optimize_parser = subparsers.add_parser("optimize", help="Run inverse design optimization.")
    optimize_parser.add_argument("--model-path", default="models/best_surrogate.pth")
    optimize_parser.add_argument("--target-wavelength", type=float, default=None)
    optimize_parser.add_argument("--target-spectrum-path", type=str, default=None)
    optimize_parser.add_argument("--steps", type=int, default=500)
    optimize_parser.add_argument("--learning-rate", type=float, default=0.03)
    optimize_parser.add_argument("--period-penalty-weight", type=float, default=5.0)
    optimize_parser.add_argument("--seed", type=int, default=42)
    optimize_parser.add_argument("--device", type=str, default=None)
    optimize_parser.add_argument("--output-json", type=str, default="results/optimized_design.json")
    optimize_parser.set_defaults(handler=_cmd_optimize)

    verify_parser = subparsers.add_parser("verify", help="Validate optimized design with FDTD.")
    verify_parser.add_argument("--geometry", type=str, default=None, help="w1,w2,w3,g1,g2,g3 in nm")
    verify_parser.add_argument("--geometry-file", type=str, default=None, help="JSON/NPY geometry file")
    verify_parser.add_argument("--model-path", type=str, default="models/best_surrogate.pth")
    verify_parser.add_argument("--target-wavelength", type=float, default=None)
    verify_parser.add_argument("--target-spectrum-path", type=str, default=None)
    verify_parser.add_argument("--device", type=str, default=None)
    verify_parser.add_argument("--output-plot", type=str, default="results/validation_plot.png")
    verify_parser.add_argument("--output-json", type=str, default="results/validation_summary.json")
    verify_parser.set_defaults(handler=_cmd_verify)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

