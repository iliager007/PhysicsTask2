"""Category 2 — Floor & Ball Collisions (3-D).

Cloth is built in the horizontal X-Z plane at a height well above the
floor (y = 0) so it falls under gravity, demonstrating bounce, friction,
and self-collision.
"""
import numpy as np
from .base import BaseScene


class FreeFallScene(BaseScene):
    @property
    def name(self):
        return "Free fall to floor"

    @property
    def category(self):
        return "2 · Floor & Ball Collisions"

    @property
    def default_self_collision(self):
        return True

    def setup(self, sim):
        sim._build_cloth(10, 16, 20, (0, 320, 0), "none", "xz")
        sim.ball_active = False
        sim.camera.target = np.array([0.0, 120.0, 0.0])
        sim.camera.distance = 800


class DropOntoBallScene(BaseScene):
    @property
    def name(self):
        return "Drop onto ball"

    @property
    def category(self):
        return "2 · Floor & Ball Collisions"

    @property
    def default_self_collision(self):
        return True

    def setup(self, sim):
        sim._build_cloth(10, 14, 20, (0, 320, 0), "none", "xz")
        sim.ball_active = True
        sim.ball_pos = np.array([0.0, 120.0, 0.0])
        sim.ball_radius = 65.0
        sim.camera.target = np.array([0.0, 120.0, 0.0])
        sim.camera.distance = 800


class DenseCrumpleScene(BaseScene):
    @property
    def name(self):
        return "Dense crumple (self-col)"

    @property
    def category(self):
        return "2 · Floor & Ball Collisions"

    @property
    def default_self_collision(self):
        return True

    def setup(self, sim):
        sim._build_cloth(10, 10, 16, (0, 280, 0), "none", "xz")
        sim.ball_active = True
        sim.ball_pos = np.array([0.0, 40.0, 0.0])
        sim.ball_radius = 40.0
        sim.camera.target = np.array([0.0, 100.0, 0.0])
        sim.camera.distance = 700
