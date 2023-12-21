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
