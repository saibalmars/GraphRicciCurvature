try:
    from ._version import version as __version__
except ImportError:
    # The generated file is added during packaging. This fallback keeps a
    # source checkout importable before the package has been built.
    __version__ = "0+unknown"

__author__ = "Chien-Chun Ni"
__email__ = "saibalmars@gmail.com"
