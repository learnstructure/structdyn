"""
structdyn: A Python library for solving structural dynamics problems numerically.

Author: Abinash Mandal

"""

# Core classes
from .sdf.sdf import SDF
from .mdf.mdf import MDF
from .ground_motions.ground_motion import GroundMotion
from .sdf.response_spectrum import ResponseSpectrum

# SDF Numerical Methods
from .sdf.numerical_methods.central_difference import CentralDifference
from .sdf.numerical_methods.interpolation import Interpolation
from .sdf.numerical_methods.newmark_beta import NewmarkBeta

# SDF Analytical Methods
from .sdf.analytical_methods.analytical_response import AnalyticalResponse

# MDF Analytical Methods
from .mdf.analytical_methods.modal_analysis import ModalAnalysis
from .mdf.analytical_methods.response_spectrum_analysis import ResponseSpectrumAnalysis

# Material Models
from .utils.material_models import (
    LinearElastic,
    ElasticPerfectlyPlastic,
    Bilinear,
    BoucWen,
    RambergOsgood,
    Takeda,
)

# Helper Functions
from .utils.helpers import plot_displacement, elcentro_chopra

__all__ = [
    # Core classes
    "SDF",
    "MDF",
    "GroundMotion",
    "ResponseSpectrum",
    # SDF Numerical Methods
    "CentralDifference",
    "Interpolation",
    "NewmarkBeta",
    # SDF Analytical Methods
    "AnalyticalResponse",
    # MDF Analytical Methods
    "ModalAnalysis",
    "ResponseSpectrumAnalysis",
    # Material Models
    "LinearElastic",
    "ElasticPerfectlyPlastic",
    "Bilinear",
    "BoucWen",
    "RambergOsgood",
    "Takeda",
    # Helper Functions
    "plot_displacement",
    "elcentro_chopra",
]
