import numpy as np
import random
from typing import List, Tuple, Dict, Any
from .config import SLAMConfig, Direction, Vector
from .environment import FireScoutEnvironment

class FireScoutRobot:
    """
    Represents the robot agent.
    Handles True State (not estimated), Kinematics, and Sensing.
    """
    def __init__(self, config: SLAMConfig, env: FireScoutEnvironment):
        self.config = config
        self.env = env
        
        # True State [row, col]
        self.true_pos = np.array(config.start_pos, dtype=float).reshape(2, 1)
        self.orientation = Direction.UP
        
        # Navigation Memory
        self.path_history: List[Tuple[int, int]] = [config.start_pos]
        self.recent_path: List[Tuple[int, int]] = [] 
        
        # Internal "Visit Map" for Coverage Logic
        # Tracks how many times we visited each cell to encourage exploration
        self.visit_map = np.zeros((env.rows, env.cols), dtype=int)
        start_r, start_c = config.start_pos
        self.visit_map[start_r, start_c] = 1

    def move_smart_explorer(self) -> Vector:
        """
        Coverage-based movement. 
        Prefers unvisited cells over visited ones (Repulsive Potential Field).
        """
        curr_r = int(np.round(self.true_pos[0, 0]))
        curr_c = int(np.round(self.true_pos[1, 0]))
        
        # 1. Identify Candidate Moves (Up, Right, Down, Left)
        # (dr, dc, direction_enum)
        candidates = [
            (-1, 0, Direction.UP),
            (0, 1, Direction.RIGHT),
            (1, 0, Direction.DOWN),
            (0, -1, Direction.LEFT)
        ]
        
        possible_moves = []
        
        for dr, dc, direct in candidates:
            next_r, next_c = curr_r + dr, curr_c + dc
            
            # Check Physical Validity (Walls/Bounds)
            if self.env.is_valid_location(next_r, next_c):
                # Calculate Cost Function
                # Cost = (Visit Count * 10) + (Turn Penalty)
                
                visits = self.visit_map[next_r, next_c]
                
                # Turn Penalty: penalize changing direction slightly to prevent jitter
                turn_cost = 0 if direct == self.orientation else 1.5
                
                # Random noise breaks symmetry/loops
                noise = random.uniform(0, 0.5)
                
                score = (visits * 10) + turn_cost + noise
                
                possible_moves.append({
                    'move': (dr, dc),
                    'score': score,
                    'orientation': direct,
                    'pos': (next_r, next_c)
                })
        
        # 2. Select Best Move
        if not possible_moves:
            # Trapped? Stay put
            next_move = (0, 0)
        else:
            # Sort by score (lowest cost is best)
            possible_moves.sort(key=lambda x: x['score'])
            best = possible_moves[0]
            
            next_move = best['move']
            self.orientation = best['orientation']
            
            # Update Visit Map
            target_r, target_c = best['pos']
            self.visit_map[target_r, target_c] += 1

        # 3. Execute Move
        self.true_pos += np.array(next_move).reshape(2, 1)
        
        # Update History
        new_r = int(np.round(self.true_pos[0, 0]))
        new_c = int(np.round(self.true_pos[1, 0]))
        
        pos_tuple = (new_r, new_c)
        self.path_history.append(pos_tuple)
        
        return np.array(next_move).reshape(2, 1)

    def sense(self) -> List[Dict[str, Any]]:
        """
        Simulates LIDAR/Visual sensor. 
        Returns list of observations: {'type': str, 'global_pos': np.array}
        """
        observations = []
        
        r_center = int(np.round(self.true_pos[0, 0]))
        c_center = int(np.round(self.true_pos[1, 0]))
        
        rad = self.config.fov_radius
        
        for r in range(r_center - rad, r_center + rad + 1):
            for c in range(c_center - rad, c_center + rad + 1):
                if 0 <= r < self.env.rows and 0 <= c < self.env.cols:
                    feat = self.env.grid[r, c]
                    if feat in [1, 2, 3]:
                        ftype = 'wall' if feat == 1 else ('debris' if feat == 2 else 'hotspot')
                        observations.append({
                            'type': ftype,
                            'global_pos': np.array([[r], [c]], dtype=float)
                        })
        return observations