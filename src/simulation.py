import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np
from tqdm.notebook import tqdm
import math
from IPython.display import display, Image

from .config import SLAMConfig
from .environment import FireScoutEnvironment
from .robot import FireScoutRobot
from .ekf import EKFEstimator

class FireScoutSimulation:
    def __init__(self, config: SLAMConfig):
        self.config = config
        self.env = FireScoutEnvironment(config)
        self.robot = FireScoutRobot(config, self.env)
        self.ekf = EKFEstimator(config)
        
        # Stats & History
        self.time_elapsed = 0.0
        self.mapping_complete = False
        self.mapped_cells = set()
        
        # History Log for Animation: Stores dictionaries of state at each step
        self.history = []

    def run(self):
        print(f"Starting Simulation: {self.config.grid_rows}x{self.config.grid_cols} Grid")
        self._log_state()
        
        for step in tqdm(range(self.config.num_steps), desc="Simulating"):
            if self.mapping_complete:
                print(f"✅ Mission Success! All targets found in {step} steps.")
                break
                
            # 1. Move (SWITCHED TO SMART EXPLORER)
            u = self.robot.move_smart_explorer()
            
            # 2. Predict
            self.ekf.predict(u)
            
            # 3. Sense
            observations = self.robot.sense()
            
            # 4. Update
            self.ekf.update(observations)
            
            # 5. Check Completion
            self._check_completion()
            self._update_mapped_coverage()
            self.time_elapsed += self.config.time_per_step
            
            # 6. Log State
            self._log_state()

    def _log_state(self):
        """Snapshots the current system state for replay."""
        # Deep copy needed for mutable numpy arrays
        snapshot = {
            'robot_true': self.robot.true_pos.copy(),
            'ekf_mu': self.ekf.mu.copy(),
            'ekf_landmarks': self.ekf.landmark_registry.copy(), # Dict copy
            'mapped': self.mapped_cells.copy()
        }
        self.history.append(snapshot)

    def _check_completion(self):
        visited_count = 0
        for tx, ty in self.env.targets:
            for (rx, ry) in self.robot.path_history:
                dist = math.sqrt((tx - rx)**2 + (ty - ry)**2)
                if dist <= self.config.target_tolerance:
                    visited_count += 1
                    break
        
        if visited_count == len(self.env.targets):
            self.mapping_complete = True

    def _update_mapped_coverage(self):
        # Mark cells in FOV as mapped
        r = int(self.robot.true_pos[0, 0])
        c = int(self.robot.true_pos[1, 0])
        
        rad = self.config.fov_radius
        for i in range(r - rad, r + rad + 1):
            for j in range(c - rad, c + rad + 1):
                if 0 <= i < self.env.rows and 0 <= j < self.env.cols:
                    self.mapped_cells.add((i, j))

    def save_animation(self, filename="mission_replay.gif", stride=2):
        """
        Generates and saves a GIF of the mission.
        stride: Plot every Nth frame to save time/space.
        """
        print(f"Generating Animation ({len(self.history)} frames)...")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 1. Static Background (Grid)
        display_grid = np.zeros_like(self.env.grid)
        display_grid[self.env.grid == 1] = 1 # Walls
        ax.imshow(display_grid, cmap='Greys', origin='upper')
        
        # Plot Static True Hazards/Hotspots
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                feat = self.env.grid[r, c]
                if feat == 2:
                    ax.scatter(c, r, c='orange', marker='^', alpha=0.5)
                elif feat == 3:
                    ax.scatter(c, r, c='red', marker='*', alpha=0.5)
        
        # Plot Targets
        for tx, ty in self.env.targets:
            circle = plt.Circle((ty, tx), self.config.target_tolerance, color='cyan', fill=False, linestyle='--')
            ax.add_patch(circle)
        
        # 2. Dynamic Artists (References to update later)
        robot_true_dot, = ax.plot([], [], 'lime', marker='o', markersize=6, label='True Pos')
        robot_est_dot, = ax.plot([], [], 'blue', marker='o', markersize=6, alpha=0.7, label='EKF Est')
        # We use a scatter plot container for landmarks as their number changes
        landmark_scatter = ax.scatter([], [], c='purple', marker='x', s=80, label='Mapped LMs')
        
        path_line, = ax.plot([], [], 'lime', linewidth=1, alpha=0.5)
        
        ax.legend(loc='upper right')
        ax.set_title("Initializing...")

        # 3. Update Function
        def update(frame_idx):
            state = self.history[frame_idx]
            
            # Update Robot True
            rx, ry = state['robot_true'][0, 0], state['robot_true'][1, 0]
            robot_true_dot.set_data([ry], [rx]) # Note: Plot uses (x=col, y=row)
            
            # Update Path Trail (Reconstruct from history up to this frame)
            # For speed, we just take every previous frame's true pos
            # In a huge sim, we might optimize this, but for <1000 frames it's fine
            past_states = self.history[:frame_idx+1:stride] 
            path_y = [s['robot_true'][1, 0] for s in past_states]
            path_x = [s['robot_true'][0, 0] for s in past_states]
            path_line.set_data(path_y, path_x)
            
            # Update Robot Estimate
            ex, ey = state['ekf_mu'][0], state['ekf_mu'][1]
            robot_est_dot.set_data([ey], [ex])
            
            # Update Landmarks
            # Extract landmarks from EKF state using the registry at that time
            lm_x = []
            lm_y = []
            for lm_id, idx in state['ekf_landmarks'].items():
                if idx+1 < len(state['ekf_mu']):
                    lm_x.append(state['ekf_mu'][idx])
                    lm_y.append(state['ekf_mu'][idx+1])
            
            # Scatter requires an array of offsets (N, 2) -> (col, row)
            if lm_x:
                offsets = np.column_stack((lm_y, lm_x))
                landmark_scatter.set_offsets(offsets)
            
            ax.set_title(f"Step {frame_idx} | Mapped LMs: {len(lm_x)}")
            return robot_true_dot, robot_est_dot, landmark_scatter, path_line

        # 4. Create Animation
        # Reduce total frames by striding
        frames = range(0, len(self.history), stride)
        ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=50)
        
        # 5. Save
        ani.save(filename, writer=PillowWriter(fps=20))
        plt.close(fig) # Prevent double plotting
        print(f"Animation saved to {filename}")
        
        # Display in Notebook
        return Image(filename)