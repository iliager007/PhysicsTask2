"""Extended Position-Based Dynamics solver (3-D) — Müller et al. / Hitman-style."""
import numpy as np

from .base import BaseSolver
from constants import DT, GRAVITY, FLOOR_Y


class XPBDSolver(BaseSolver):

    @property
    def name(self) -> str:
        return "PBD / XPBD"

    def step(self, sim) -> None:
        dt = DT
        mask = sim.inv_mass > 0

        pre_vel_y = sim.vel[:, 1].copy()
        was_above = sim.pos[:, 1] > FLOOR_Y + 1.0

        sim.vel[mask] += GRAVITY * dt
        sim.vel *= sim.damping
        sim.predicted = sim.pos + sim.vel * dt

        if sim.grabbed >= 0:
            sim.predicted[sim.grabbed] = sim.mouse

        alpha = max(1e-8, (1.0 - sim.stiffness) * 0.1) / (dt * dt)

        for _ in range(sim.iterations):
            self._solve_distance(sim, alpha)
            self._enforce_pins(sim)
            self._collision_pass(sim, sim.predicted)
            if sim.grabbed >= 0:
                sim.predicted[sim.grabbed] = sim.mouse

        sim.vel[mask] = (sim.predicted[mask] - sim.pos[mask]) / dt
        sim.pos[mask] = sim.predicted[mask]
        self._post_step_bounce(sim, pre_vel_y, was_above)

    # ------------------------------------------------------------------
    def _solve_distance(self, sim, alpha: float) -> None:
        p1 = sim.predicted[sim.c_i]
        p2 = sim.predicted[sim.c_j]
        diff = p2 - p1
        dist = np.linalg.norm(diff, axis=1)
        dist_safe = np.maximum(dist, 1e-7)

        C = dist - sim.c_rest
        denom = sim.c_wsum + alpha
        cmag = np.where(denom > 0, C / denom, 0.0)

        correction = (cmag / dist_safe)[:, np.newaxis] * diff

        dp = np.zeros((sim.n, 3))
        np.add.at(dp, sim.c_i,  sim.c_wi[:, np.newaxis] * correction)
        np.add.at(dp, sim.c_j, -sim.c_wj[:, np.newaxis] * correction)
        sim.predicted += dp

    @staticmethod
    def _enforce_pins(sim) -> None:
        for i in np.where(sim.pin_mask)[0]:
            if i != sim.grabbed:
                sim.predicted[i] = sim.pinned_pos[i]
