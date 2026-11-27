"""Stable identifiers for PyECR model families."""

from enum import Enum


class ModelID(str, Enum):
    """Model identifiers used by configuration, fitting, and reporting."""

    M0_SINGLE_D = "M0_SINGLE_D"
    M1_SINGLE_DK = "M1_SINGLE_DK"
    M2_DUAL_HO_D = "M2_DUAL_HO_D"
    M3_DUAL_HO_DK = "M3_DUAL_HO_DK"
    M4_FLUSH = "M4_FLUSH"
    M5_GLOBAL = "M5_GLOBAL"
    M6_DCT = "M6_DCT"
    M7_SINGLE_K = "M7_SINGLE_K"
    M8_DUAL_K = "M8_DUAL_K"

