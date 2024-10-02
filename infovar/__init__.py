import importlib.metadata

__version__ = importlib.metadata.version("infovar")

from .handlers import *
from .stats import *
from .processing import *

__all__ = []
__all__.extend(handlers.__all__)
__all__.extend(stats.__all__)
__all__.extend(processing.__all__)
