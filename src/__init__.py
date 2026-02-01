"""
FireScout SLAM Simulation Package
Exposes core classes for easy import.
"""

# Relative imports allow moving the 'src' folder without breaking links
from .config import SLAMConfig, Direction
from .environment import FireScoutEnvironment
from .robot import FireScoutRobot
from .ekf import EKFEstimator
from .simulation import FireScoutSimulation

# __all__ defines what gets imported if someone uses 'from src import *'
# It acts as a strict definition of your Public API.
__all__ = [
    'SLAMConfig',
    'Direction',
    'FireScoutEnvironment',
    'FireScoutRobot',
    'EKFEstimator',
    'FireScoutSimulation',
]