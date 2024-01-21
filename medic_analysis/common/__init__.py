"""Base functions used by the scripts in medic_analysis/scripts."""
from memori.logging import setup_logging

from medic_analysis.common import align, bias_field, distortion, figures

__all__ = [
    "align",
    "bias_field",
    "distortion",
    "figures",
]

# setup the logger on module import
setup_logging()
