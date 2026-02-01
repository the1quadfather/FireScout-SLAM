import numpy as np
from typing import List, Tuple, Dict
from .config import SLAMConfig, Vector  # Relative import

class EKFEstimator:
    """
    Extended Kalman Filter for SLAM.
    State Vector mu: [robot_x, robot_y, landmark1_x, landmark1_y, ...]
    """
    def __init__(self, config: SLAMConfig):
        self.config = config
        
        # Initial Robot State (2x1)
        self.mu = np.array(config.start_pos, dtype=float).reshape(2, 1)
        self.cov = config.initial_cov.copy()
        
        # Landmark Management
        # Maps (row, col) tuple to index in the state vector
        self.landmark_registry: Dict[Tuple[int, int], int] = {} 
        self.num_landmarks = 0

    def predict(self, u: Vector):
        """
        Motion Model Prediction Step.
        x_t = F * x_{t-1} + B * u_t
        """
        # 1. Update Robot State
        # In this linear grid model, prediction is simple addition
        self.mu[:2] += u
        
        # 2. Update Covariance
        # We need to add process noise Q only to the robot's 2x2 block
        # The landmarks are static, so their process noise is 0.
        
        # Expand Q to match full state size? 
        # No, technically we generate a F_x matrix to map Q to the full state space.
        # P = F P F^T + F_x Q F_x^T
        # Since our motion model is linear and identity based:
        # P_robot_block += Q
        
        self.cov[:2, :2] += self.config.Q

    def update(self, observations: List[Dict]):
        """
        Measurement Model Update Step.
        Corrects state based on observations.
        """
        for obs in observations:
            if obs['type'] == 'wall':
                self._update_known_feature(obs, self.config.R_wall)
            else:
                self._update_landmark(obs, self.config.R_landmark)

    def _update_known_feature(self, obs, R):
        """Standard KF update for non-state features (like walls used for localization only)."""
        # Note: In pure SLAM, walls usually aren't landmarks unless mapped.
        # Here we treat walls as known map features to aid localization (Localization only).
        z = obs['global_pos'] # Observed position
        
        # Expected measurement (h(x)) is just the robot pos (if relative)
        # But here 'z' is global grid pos. 
        # Innovation = Observed_Global - Predicted_Robot_Global (Simplification for grid)
        # Actually, if we see a wall at (10,10), and we think we are at (10,9),
        # Relative measurement is (0, 1). 
        
        # Simplified update: We treat the wall observation as a direct observation of position 
        # relative to the wall. 
        # z_t = H * x_t + v
        # Here we use the direct observation logic from the original notebook
        
        # H matrix for robot state (2x2 identity for position)
        H = np.eye(2) 
        
        # Innovation y = z - Hx (but conceptually applied to alignment)
        # In the provided code, wall updates helped localize the robot.
        # We will skip wall updates for 'mapping' and purely use them to reduce robot Cov
        # if the logic allows. For simplicity in this refactor, we stick to Landmarks for SLAM.
        pass

    def _update_landmark(self, obs, R):
        """EKF Update for Landmarks (Debris/Hotspots)."""
        z_global = obs['global_pos']
        lm_id = tuple(np.round(z_global).flatten().astype(int))
        
        # 1. Landmark Initialization (if new)
        if lm_id not in self.landmark_registry:
            self._register_new_landmark(z_global, lm_id)
            
        # 2. EKF Update
        # Get landmark index in state vector
        lm_idx = self.landmark_registry[lm_id]
        
        # Extract landmark estimate from state
        lm_pos = self.mu[lm_idx:lm_idx+2]
        robot_pos = self.mu[0:2]
        
        # Expected relative measurement
        z_expected = lm_pos - robot_pos 
        
        # Actual relative measurement (Observed Global - Current Robot Estimate)
        # Note: In real SLAM, sensors give relative (Range/Bearing). 
        # Here the "sensor" gives global coords based on true pos.
        # So Actual_Relative = Observed_Global - Robot_Estimate
        z_actual_rel = z_global - robot_pos
        
        y = z_actual_rel - z_expected # Innovation
        
        # Jacobian H construction
        # H has shape (2, State_Dim). 
        # It has -I at robot index and +I at landmark index
        H = np.zeros((2, len(self.mu)))
        H[:, 0:2] = -np.eye(2)
        H[:, lm_idx:lm_idx+2] = np.eye(2)
        
        # Kalman Gain
        # K = P H^T (H P H^T + R)^-1
        PHt = self.cov @ H.T
        S = H @ PHt + R
        K = PHt @ np.linalg.inv(S)
        
        # State Update
        self.mu = self.mu + K @ y
        
        # Covariance Update
        # P = (I - K H) P
        I = np.eye(len(self.mu))
        self.cov = (I - K @ H) @ self.cov

    def _register_new_landmark(self, pos: Vector, lm_id: Tuple[int, int]):
        """Expands State and Covariance matrices for new landmark."""
        self.num_landmarks += 1
        idx = 2 + (self.num_landmarks - 1) * 2
        self.landmark_registry[lm_id] = idx
        
        # Extend State Vector
        self.mu = np.vstack([self.mu, pos])
        
        # Extend Covariance Matrix
        # New dimensions
        old_dim = self.cov.shape[0]
        new_dim = old_dim + 2
        new_cov = np.zeros((new_dim, new_dim))
        
        # Copy old cov
        new_cov[:old_dim, :old_dim] = self.cov
        
        # Initialize new landmark covariance (high uncertainty initially)
        # Or based on sensor noise + robot uncertainty
        new_cov[old_dim:, old_dim:] = np.eye(2) * 1.0 # Initial uncertainty
        
        self.cov = new_cov