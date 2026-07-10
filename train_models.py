from __future__ import annotations

import json
from pathlib import Path

from src.titanic_pipeline import REPORTS_DIR, train_and_evaluate


def main() -> None:
    results = train_and_evaluate()
    serializable = {
        "classification": results["classification"],
        "regression": results["regression"],
        "rows": results["rows"],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(REPORTS_DIR) / "metrics.json"
    metrics_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"Modelos entrenados. Metricas guardadas en {metrics_path}")


if __name__ == "__main__":
    main()
