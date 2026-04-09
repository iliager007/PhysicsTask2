"""Category 1 — Hanging from Fixed Points (3-D).

Cloth is built in the vertical X-Y plane so that gravity (-Y) pulls it
downward while the Z axis is available for out-of-plane deformation
around the ball.
"""
import numpy as np
from .base import BaseScene


class SinglePointScene(BaseScene):
    @property
    def name(self):
        return "Single center point"

    @property
    def category(self):
        return "1 · Hanging from Fixed Points"

    def setup(self, sim):
        sim._build_cloth(14, 14, 20, (0, 150, 0), "single_center", "xy")
        sim.ball_active = True
        sim.ball_pos = np.array([0.0, 30.0, 80.0])
        sim.ball_radius = 55.0
        sim.camera.target = np.array([0.0, 80.0, 0.0])
        sim.camera.distance = 750


class TwoCornersScene(BaseScene):
    @property
    def name(self):
        return "Two top corners"

    @property
    def category(self):
        return "1 · Hanging from Fixed Points"

    def setup(self, sim):
        sim._build_cloth(14, 20, 20, (0, 150, 0), "corners", "xy")
        sim.ball_active = True
        sim.ball_pos = np.array([0.0, 50.0, 80.0])
        sim.ball_radius = 55.0
        sim.camera.target = np.array([0.0, 80.0, 0.0])
        sim.camera.distance = 800


class ThreePointsScene(BaseScene):
    @property
    def name(self):
        return "Three spread points"

    @property
    def category(self):
        return "1 · Hanging from Fixed Points"

    def setup(self, sim):
        sim._build_cloth(12, 22, 18, (0, 140, 0), "three_top", "xy")
        sim.ball_active = True
        sim.ball_pos = np.array([0.0, 50.0, 70.0])
        sim.ball_radius = 50.0
        sim.camera.target = np.array([0.0, 70.0, 0.0])
        sim.camera.distance = 800


class FullTopEdgeScene(BaseScene):
    @property
    def name(self):
        return "Full top edge"

    @property
    def category(self):
        return "1 · Hanging from Fixed Points"

    def setup(self, sim):
        sim._build_cloth(15, 20, 20, (0, 160, 0), "top", "xy")
        sim.ball_active = True
        sim.ball_pos = np.array([0.0, 60.0, 80.0])
        sim.ball_radius = 60.0
        sim.camera.target = np.array([0.0, 80.0, 0.0])
        sim.camera.distance = 800
