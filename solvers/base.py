"""Abstract base class for all cloth physics solvers.

Includes shared collision utilities that every solver reuses.
"""
import numpy as np
from abc import ABC, abstractmethod
from constants import FLOOR_Y


class BaseSolver(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def implemented(self) -> bool:
        return True

    @abstractmethod
    def step(self, sim) -> None:
        ...

    # ------------------------------------------------------------------
    # Shared collision helpers — operate on an arbitrary position array
    # so each solver can call them on sim.predicted, a local q, etc.
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_floor(sim, q):
        below = q[:, 1] < FLOOR_Y
        if not np.any(below):
            return
        q[below, 1] = FLOOR_Y
        fric = sim.floor_friction * 0.5
        for axis in (0, 2):
            d = q[below, axis] - sim.pos[below, axis]
            q[below, axis] = sim.pos[below, axis] + d * (1.0 - fric)

    @staticmethod
    def _clamp_ball(sim, q):
        if not sim.ball_active:
            return
        d = q - sim.ball_pos
        dn = np.linalg.norm(d, axis=1)
        mindist = sim.ball_radius + sim.particle_r
        hit = dn < mindist
        if not np.any(hit):
            return
        dns = np.maximum(dn[hit], 1e-7)
        norm = d[hit] / dns[:, np.newaxis]
        q[hit] = sim.ball_pos + norm * mindist

    @staticmethod
    def _clamp_ball_safe(sim, q):
        """Ball collision with wrap-through prevention.

        Uses the current position direction for normal surface contact
        (allows natural sliding), but falls back to the previous-frame
        direction when the particle has clearly passed through the ball
        (dot product between current and previous directions is negative).
        """
        if not sim.ball_active:
            return
        d = q - sim.ball_pos
        dn = np.linalg.norm(d, axis=1)
        mindist = sim.ball_radius + sim.particle_r
        hit = dn < mindist
        if not np.any(hit):
            return

        d_cur = d[hit]
        dns = np.maximum(dn[hit], 1e-7)
        norm = d_cur / dns[:, np.newaxis]

        d_prev = sim.pos[hit] - sim.ball_pos
        dn_prev = np.linalg.norm(d_prev, axis=1)
        has_prev = dn_prev > 1e-4

        # Detect wrap-through: particle is on the opposite side of the ball
        dot = np.sum(d_cur * d_prev, axis=1)
        passed_through = (dot < 0) & has_prev

        if np.any(passed_through):
            norm[passed_through] = (d_prev[passed_through]
                                    / dn_prev[passed_through, np.newaxis])

        q[hit] = sim.ball_pos + norm * mindist

    @staticmethod
    def _self_collision(sim, q):
        min_d = sim.particle_r * 5
        cell = min_d * 2
        grid: dict = {}
        for i in range(sim.n):
            key = (int(q[i, 0] / cell),
                   int(q[i, 1] / cell),
                   int(q[i, 2] / cell))
            if key not in grid:
                grid[key] = []
            grid[key].append(i)

        cols = sim.cols
        for pts in grid.values():
            cx = int(q[pts[0], 0] / cell)
            cy = int(q[pts[0], 1] / cell)
            cz = int(q[pts[0], 2] / cell)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        npts = grid.get((cx + dx, cy + dy, cz + dz))
                        if npts is None:
                            continue
                        for i in pts:
                            for j in npts:
                                if j <= i:
                                    continue
                                ri, ci_c = divmod(i, cols)
                                rj, cj_c = divmod(j, cols)
                                if abs(ri - rj) <= 1 and abs(ci_c - cj_c) <= 1:
                                    continue
                                diff = q[j] - q[i]
                                dist = np.linalg.norm(diff)
                                if dist < min_d and dist > 1e-7:
                                    n = diff / dist
                                    corr = (min_d - dist) * 0.5
                                    wi, wj = sim.inv_mass[i], sim.inv_mass[j]
                                    ws = wi + wj
                                    if ws > 0:
                                        q[i] -= n * corr * (wi / ws)
                                        q[j] += n * corr * (wj / ws)

    @staticmethod
    def _post_step_bounce(sim, pre_vel_y, was_above):
        on_floor = sim.pos[:, 1] <= FLOOR_Y + 0.5
        just_landed = on_floor & was_above
        if np.any(just_landed):
            sim.vel[just_landed, 1] = np.abs(pre_vel_y[just_landed]) * sim.restitution
        if np.any(on_floor):
            fric = 1.0 - sim.floor_friction * 0.4
            sim.vel[on_floor, 0] *= fric
            sim.vel[on_floor, 2] *= fric

    def _collision_pass(self, sim, q):
        """Run all collision projections on *q*."""
        self._clamp_floor(sim, q)
        self._clamp_ball(sim, q)
        if sim.self_collision:
            self._self_collision(sim, q)
