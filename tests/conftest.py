import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pytest_addoption(parser):
    parser.addoption(
        "--run-live", action="store_true", default=False,
        help="run tests marked 'live' (hit real LunaRoute/emulator; spends credits)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: hits real LunaRoute/emulator; skipped unless RUN_LIVE=1 or --run-live",
    )


def pytest_collection_modifyitems(config, items):
    import pytest

    if config.getoption("--run-live") or os.environ.get("RUN_LIVE") == "1":
        return
    skip_live = pytest.mark.skip(reason="live test; set RUN_LIVE=1 or --run-live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
