from pathlib import Path

ROOT_DIR = Path("./").resolve()

ASSETS_DIR = ROOT_DIR / "assets"
# NOTE wandb adds a '/wandb' postfix to its run dir, so WANDB_DIR refers to
# a directory created by passing ROOT_DIR as dir to wandb.init
WANDB_DIR = ASSETS_DIR / "wandb"
TENSORFLOW_LOGS_DIR = ASSETS_DIR / "tensorflow_runs"
DEMONSTRATIONS_DIR = ASSETS_DIR / "demonstrators"

TEST_DIR = ROOT_DIR / "tests"
TEST_ASSETS_DIR = TEST_DIR / "assets"
