
from . import _version
__version__ = _version.get_versions()['version']

try:
    from numba import njit
    has_numba = True
except ImportError:
    has_numba = False

__all__ = [
    '__version__',
    has_numba
]
