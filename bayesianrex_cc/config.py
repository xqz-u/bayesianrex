from pathlib import Path

# NOTE this is relative to where the Python interpreter is run (so the root of
# the repo)
ROOT = Path("./").resolve()

CODE_DIR = ROOT / "bayesianrex_cc"

LOGS_DIR = ROOT / "logs"
WANDB_LOGS_DIR = ROOT
TB_LOGS_DIR = ROOT / "runs"
DATA_DIR = ROOT / "data"
