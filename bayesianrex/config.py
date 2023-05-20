from pathlib import Path

ROOT_DIR = Path("./").resolve()

# NOTE wandb adds a '/wandb' postfix to its run dir, so WANDB_DIR refers to
# a directory created by passing ROOT_DIR as dir to wandb.init
WANDB_DIR = ROOT_DIR / "wandb"
TENSORFLOW_LOGS_DIR = ROOT_DIR / "tensorflow_runs"

TEST_DIR = ROOT_DIR / "tests"
TEST_ASSETS_DIR = TEST_DIR / "assets"

DATA_DIR = ROOT_DIR / "data"
DEMONSTRATIONS_DIR = DATA_DIR / "demonstrations"
