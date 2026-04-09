"""Shared constants used across simulation, solvers and scenes."""
import numpy as np

W: int = 800
H: int = 600
FLOOR_Y: float = 0.0          # y-up; floor is the y = 0 plane
GRAVITY = np.array([0.0, -980.0, 0.0])
DT: float = 1.0 / 60.0
TAU: float = 2.0 * np.pi
