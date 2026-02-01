import numpy as np
import random
from .config import SLAMConfig

class FireScoutEnvironment:
    """
    Represents the physical world (Ground Truth).
    Manages walls, hazards, and hotspots.
    """
    def __init__(self, config: SLAMConfig):
        self.config = config
        self.rows = config.grid_rows
        self.cols = config.grid_cols
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.targets = [(1, 1), (1, 18), (18, 1), (18, 18), (10, 10)]
        self._initialize_grid()

    def _initialize_grid(self):
        """Sets up walls, hazards, and hotspots."""
        # 1. Place Walls (Perimeter)
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # 2. Place Hazards (2) and Hotspots (3)
        self._place_features(self.config.num_hazards, 2)
        self._place_features(self.config.num_hotspots, 3)

    def _place_features(self, count: int, feature_id: int):
        """Randomly places features avoiding start pos and existing features."""
        placed = 0
        while placed < count:
            r = random.randint(1, self.rows - 2)
            c = random.randint(1, self.cols - 2)
            # Avoid start position and walls/existing features
            start_r, start_c = self.config.start_pos
            if self.grid[r, c] == 0 and (r, c) != (start_r, start_c):
                self.grid[r, c] = feature_id
                placed += 1

    def get_feature_at(self, row: int, col: int) -> int:
        """Safe grid access with bounds checking."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row, col]
        return 1 # Treat out of bounds as wall

    def is_valid_location(self, row: int, col: int) -> bool:
        """Check if a location is traversable (not a wall)."""
        return 0 <= row < self.rows and 0 <= col < self.cols and self.grid[row, col] != 1