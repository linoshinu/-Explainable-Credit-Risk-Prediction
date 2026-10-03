"""Train the demo model suite and save a local best-model artifact."""

from __future__ import annotations

from pathlib import Path

import joblib

from credit_risk.data import FEATURES
from credit_risk.modeling import train_model_suite


def main() -> None:
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    suite = train_model_suite()
    best_name = str(suite.metrics.iloc[0]["Model"])
    joblib.dump(
        {"model_name": best_name, "model": suite.models[best_name], "features": FEATURES},
        output_dir / "best_model.joblib",
    )
    suite.metrics.to_json(output_dir / "holdout_metrics.json", orient="records", indent=2)
    print(f"Saved {best_name} to {output_dir / 'best_model.joblib'}")
    print(suite.metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
