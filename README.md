# FireScout: EKF-SLAM Robotics Simulation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**FireScout** is a modular Python simulation demonstrating **Simultaneous Localization and Mapping (SLAM)** using an Extended Kalman Filter (EKF). 

Designed as a portfolio project to showcase **Object-Oriented Architecture**, **State Estimation**, and **Autonomous Navigation**.

## Key Features

* **Extended Kalman Filter (EKF):** Implements the full predict-update cycle with Jacobian linearization for landmarks.
* **Modular Architecture:** strictly decoupled `Robot` (Kinematics), `Environment` (Ground Truth), and `EKF` (Estimation) modules.
* **Wall-Following Logic:** Autonomous navigation controller using simulated LIDAR-like sensor data.
* **Visual Analytics:** Generates post-mission mapping analysis comparing Ground Truth vs. EKF Estimation.

## 🛠️ Project Structure

```text
FireScout-SLAM/
├── main.ipynb            # Interactive Notebook & Simulation
├── README.md             # Documentation
└── src/                  # Source Package
    ├── config.py         # Configuration & Hyperparameters
    ├── ekf.py            # EKF Math & Matrix Operations
    ├── environment.py    # Map Generation & Feature Management
    ├── robot.py          # Robot Physics & Sensor Simulation
    └── simulation.py     # Orchestrator & Visualization


QUICK START
1. Clone the repo
git clone [https://github.com/the1quadfather/FireScout-SLAM.git](https://github.com/the1quadfather/FireScout-SLAM.git)
cd FireScout-SLAM

2. Install dependencies
pip install numpy matplotlib tqdm

3. Run!
Open main.ipynb and run all the cells! The parmeters in Cell 2 ("Config") can be changed to suit your curiosity.
python -m src.simulation


RESULTS
The simulation produces a GIF and a map showing:
- Green Line: Robot's true path
- Blue Dots: EKF's estimated position (should track very closely to the green line)
- Purple X: EKF's estimated landmark positions (fire, hazards, etc.)
- Orange & Red Icons: Actual landmarks


THEORY EXPLAINED
- Motion model is based on a differntial drive's kinematics with Gaussian noise.
- Observation model is a range-bearing feature extraction method
- Landmark association is done via nearest-neighbor feature matching