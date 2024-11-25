"""For testing netneurotools.plotting.pyvista_plotters functionality."""

import pytest


@pytest.mark.pyvista
def test_pyvista_smoke():
    """Test that pyvista is importable."""
    import pyvista  # noqa: F401
