"""Category 3 — Overconstrained / Incompatible Constraints (3-D).

Pin positions are moved to locations that violate the cloth's rest-length
topology, forcing the solver into a best-effort compromise.
"""
import numpy as np
from .base import BaseScene


class TwoDistantCornersScene(BaseScene):
    @property
    def name(self):
        return "Two distant corners"

    @property
    def category(self):
        return "3 · Overconstrained Stretch"

    def setup(self, sim):
        sp = 20
        sim._build_cloth(8, 20, sp, (0, 180, 0), "corners", "xy")
        nat_w = (sim.cols - 1) * sp
        target_w = nat_w * 1.8
        half_h = (sim.rows - 1) * sp / 2
        top_y = 180 + half_h
        sim.pinned_pos[0]            = [-target_w / 2, top_y, 0]
        sim.pinned_pos[sim.cols - 1] = [ target_w / 2, top_y, 0]
        sim._snap_pins()
        sim.ball_active = False
        sim.camera.target = np.array([0.0, 120.0, 0.0])
        sim.camera.distance = 800


class FourCornersSpreadScene(BaseScene):
    @property
    def name(self):
        return "Four corners spread"

    @property
    def category(self):
        return "3 · Overconstrained Stretch"

    def setup(self, sim):
        sp = 20
        sim._build_cloth(12, 12, sp, (0, 160, 0), "four_corners", "xy")
        nw = (sim.cols - 1) * sp
        nh = (sim.rows - 1) * sp
        spread = 1.7
        hw = nw * spread / 2
        hh = nh * spread / 2
        cy = 160.0
        c = sim.cols
        sim.pinned_pos[0]                            = [-hw, cy + hh, 0]
        sim.pinned_pos[c - 1]                        = [ hw, cy + hh, 0]
        sim.pinned_pos[(sim.rows - 1) * c]           = [-hw, cy - hh, 0]
        sim.pinned_pos[(sim.rows - 1) * c + c - 1]  = [ hw, cy - hh, 0]
        sim._snap_pins()
        sim.ball_active = False
        sim.camera.target = np.array([0.0, 160.0, 0.0])
        sim.camera.distance = 850


class CompressedEdgesScene(BaseScene):
    @property
    def name(self):
        return "Compressed edges"

    @property
    def category(self):
        return "3 · Overconstrained Stretch"

    @property
    def default_self_collision(self):
        return True

    def setup(self, sim):
        sp = 20
        sim._build_cloth(8, 18, sp, (0, 160, 0), "sides", "xy")
        nat_w = (sim.cols - 1) * sp
        target_w = nat_w * 0.35
        half_h = (sim.rows - 1) * sp / 2
        for r in range(sim.rows):
            y = 160.0 + half_h - r * sp
            sim.pinned_pos[r * sim.cols]               = [-target_w / 2, y, 0]
            sim.pinned_pos[r * sim.cols + sim.cols - 1] = [ target_w / 2, y, 0]
        sim._snap_pins()
        sim.ball_active = False
        sim.camera.target = np.array([0.0, 140.0, 0.0])
        sim.camera.distance = 750
