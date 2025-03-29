"""
structdyn: A Python library for solving structural dynamics problems numerically.

Author: Abinash Mandal
Version: 0.1.0
"""

from .sdf import SDF
from .central_difference import CentralDifference
from .utils import fs

__version__ = "0.1.0"
__author__ = "Abinash Mandal"

__all__ = ["SDF", "CentralDifference", "fs"]
