"""Config file for pytest fixtures."""

import pytest


def pytest_addoption(parser):
    """Add option to skip tests that fetch data."""
    parser.addoption(
        "--no_fetch",
        action="store_true",
        default=False,
        help="run tests that fetches data, could be slow"
    )


def pytest_configure(config):
    """Add markers for tests that fetch data."""
    config.addinivalue_line(
        "markers", "test_fetch: run tests that fetches data, could be slow")


def pytest_collection_modifyitems(config, items):
    """Skip tests that fetch data if --no_fetch option is used."""
    if config.getoption("--no_fetch"):
        skip_no_fetch = pytest.mark.skip(reason="remove --no_fetch option to run")
        for item in items:
            if "test_fetch" in item.keywords:
                item.add_marker(skip_no_fetch)
