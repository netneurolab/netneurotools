"""For testing netneurotools.plotting.color_utils functionality."""


def test_register_cmaps():
    """Test registering colormaps."""
    import matplotlib
    if "justine" in matplotlib.colormaps:
        assert True
    else:
        assert False
