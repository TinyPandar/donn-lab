"""Small optical operators shared by training and hardware replay."""

from .detector_psf import GaussianIntensityPSF, validate_detector_psf_sigma

__all__ = ["GaussianIntensityPSF", "validate_detector_psf_sigma"]
