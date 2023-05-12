from pathlib import Path

ROOT_DIR = Path("../")

LOGS_DIR = ROOT_DIR / "logs"
WANDB_LOGS_DIR = LOGS_DIR
TENSORFLOW_LOGS_DIR = WANDB_LOGS_DIR / "tensorflow_runs"
TENSORFLOW_LOGS_DIR.mkdir(exist_ok=True, parents=True)

TEST_DIR = ROOT_DIR / "tests"
TEST_ASSETS_DIR = TEST_DIR / "assets"
TEST_ASSETS_DIR.mkdir(exist_ok=True, parents=True)

DATA_DIR = ROOT_DIR / "data"

DEMONSTRATIONS_DIR = DATA_DIR / "demonstrations"
