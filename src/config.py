import numpy as np
import matplotlib.pyplot as plt
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from enum import IntEnum
from tqdm.notebook import tqdm

# --- Type Hints ---
# Using NewType or simple aliases helps documentation
Matrix = np.ndarray 
Vector = np.ndarray

class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

@dataclass
class SLAMConfig:
    """Central configuration for the Fire Scout Simulation."""
    # Grid Setup
    grid_rows: int = 20
    grid_cols: int = 20
    num_hazards: int = 5
    num_hotspots: int = 2
    
    # Robot Setup
    start_pos: Tuple[int, int] = (18, 1)
    fov_radius: int = 5
    memory_length: int = 20
    
    # Simulation
    num_steps: int = 500
    time_per_step: float = 0.25
    
    # EKF Hyperparameters
    # Process Noise (Motion)
    Q: Matrix = field(default_factory=lambda: np.eye(2) * 0.01)
    # Measurement Noise (Sensing Walls)
    R_wall: Matrix = field(default_factory=lambda: np.eye(2) * 0.5)
    # Measurement Noise (Sensing Landmarks)
    R_landmark: Matrix = field(default_factory=lambda: np.eye(2) * 0.1)
    # Initial State Covariance
    initial_cov: Matrix = field(default_factory=lambda: np.eye(2) * 0.1)

    # Tolerances
    target_tolerance: float = 1.5 # Radius to consider a target "visited"

print("Configuration Loaded.")