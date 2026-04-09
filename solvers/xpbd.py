"""Extended Position-Based Dynamics solver (3-D) — Müller et al. / Hitman-style."""
import numpy as np

from .base import BaseSolver
from constants import DT, GRAVITY, FLOOR_Y


class XPBDSolver(BaseSolver):

    @property
    def name(self) -> str:
        return "PBD / XPBD"

    # ------------------------------------------------------------------
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
            self._solve_floor(sim)
            if sim.ball_active:
                self._solve_ball(sim)
            if sim.self_collision:
                self._solve_self_collision(sim)
            if sim.grabbed >= 0:
                sim.predicted[sim.grabbed] = sim.mouse

        sim.vel[mask] = (sim.predicted[mask] - sim.pos[mask]) / dt
        sim.pos[mask] = sim.predicted[mask]

        # Elastic floor bounce
        on_floor = sim.pos[:, 1] <= FLOOR_Y + 0.5
        just_landed = on_floor & was_above
        if np.any(just_landed):
            sim.vel[just_landed, 1] = np.abs(pre_vel_y[just_landed]) * sim.restitution

        if np.any(on_floor):
            fric = 1.0 - sim.floor_friction * 0.4
            sim.vel[on_floor, 0] *= fric
            sim.vel[on_floor, 2] *= fric

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

    def _enforce_pins(self, sim) -> None:
        for i in np.where(sim.pin_mask)[0]:
            if i != sim.grabbed:
                sim.predicted[i] = sim.pinned_pos[i]

    def _solve_floor(self, sim) -> None:
        below = sim.predicted[:, 1] < FLOOR_Y
        if not np.any(below):
            return
        sim.predicted[below, 1] = FLOOR_Y
        fric = sim.floor_friction * 0.5
        for axis in (0, 2):
            d = sim.predicted[below, axis] - sim.pos[below, axis]
            sim.predicted[below, axis] = sim.pos[below, axis] + d * (1.0 - fric)

    def _solve_ball(self, sim) -> None:
        d = sim.predicted - sim.ball_pos
        dn = np.linalg.norm(d, axis=1)
        mindist = sim.ball_radius + sim.particle_r
        hit = dn < mindist
        if not np.any(hit):
            return
        dns = np.maximum(dn[hit], 1e-7)
        norm = d[hit] / dns[:, np.newaxis]
        sim.predicted[hit] = sim.ball_pos + norm * mindist

    # ------------------------------------------------------------------
    def _solve_self_collision(self, sim) -> None:
        min_d = sim.particle_r * 5
        cell = min_d * 2
        grid: dict = {}
        pred = sim.predicted
        for i in range(sim.n):
            key = (int(pred[i, 0] / cell),
                   int(pred[i, 1] / cell),
                   int(pred[i, 2] / cell))
            if key not in grid:
                grid[key] = []
            grid[key].append(i)

        cols = sim.cols
        for pts in grid.values():
            p0 = pts[0]
            cx = int(pred[p0, 0] / cell)
            cy = int(pred[p0, 1] / cell)
            cz = int(pred[p0, 2] / cell)
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
                                diff = pred[j] - pred[i]
                                dist = np.linalg.norm(diff)
                                if dist < min_d and dist > 1e-7:
                                    n = diff / dist
                                    corr = (min_d - dist) * 0.5
                                    wi, wj = sim.inv_mass[i], sim.inv_mass[j]
                                    ws = wi + wj
                                    if ws > 0:
                                        pred[i] -= n * corr * (wi / ws)
                                        pred[j] += n * corr * (wj / ws)
