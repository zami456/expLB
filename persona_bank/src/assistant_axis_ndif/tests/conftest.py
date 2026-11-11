"""Put the pipeline dir on sys.path so the ported modules import by bare name."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running; deselect with -m 'not slow'")
