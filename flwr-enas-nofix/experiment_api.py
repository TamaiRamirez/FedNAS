from pathlib import Path
import re

def get_experiment_dir() -> Path:
    base_dir = Path("/workspace/flwr-enas-nofix/experiments")

    if not base_dir.exists():
        raise RuntimeError(f"Base dir no existe: {base_dir}")

    pattern = re.compile(r"^(\d{8})_(\d+)$")

    candidates = []

    for p in base_dir.glob("*_*"):
        if p.is_dir():
            m = pattern.match(p.name)
            if m:
                date_part = m.group(1)
                num_part = int(m.group(2))
                candidates.append((date_part, num_part, p))

    if not candidates:
        raise RuntimeError("No se encontraron experimentos YYYYMMDD_N")

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]

def get_exp_id() -> str:
    return get_experiment_dir().name

def get_results_dir() -> Path:
    d = get_experiment_dir() / "results"
    d.mkdir(exist_ok=True)
    return d

def get_logs_dir() -> Path:
    d = get_experiment_dir() / "logs"
    d.mkdir(exist_ok=True)
    return d

def get_controller_weights_dir() -> Path:
    d = get_experiment_dir() / "controller_weights"
    d.mkdir(exist_ok=True)
    return d

def get_label_distributions_dir() -> Path:
    d = get_experiment_dir() / "label_distributions"
    d.mkdir(exist_ok=True)
    return d
