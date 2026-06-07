"""CLI placeholder for preprocessing workflows."""

from src.config import DATA_PROCESSED_DIR
from src.utils.paths import ensure_dir


def main() -> None:
    """Ensure processed data directory exists before running pipelines."""
    ensure_dir(DATA_PROCESSED_DIR)
    print(f"Processed data directory ready: {DATA_PROCESSED_DIR}")


if __name__ == "__main__":
    main()
