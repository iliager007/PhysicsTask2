"""Vertex Block Descent solver (Chen et al. 2024, SIGGRAPH).

Each iteration assembles a per-vertex 3×3 Hessian and 3-vector gradient
from the inertia term and all incident spring constraints, then solves
the local Newton step  H·Δx = −g  for every vertex in batch.

This implementation uses a **Jacobi** (simultaneous) update so the whole
solve is vectorised with NumPy.  A relaxation factor keeps it stable.

The 3×3 per-vertex solve uses Cramer's rule (pure arithmetic — no LAPACK)
so it works reliably in Pyodide/WASM.
"""
import numpy as np

from .base import BaseSolver
from constants import DT, GRAVITY, FLOOR_Y

NUM_SUBSTEPS = 2


def _solve3x3_batch(H, rhs):
    """Solve H @ x = rhs for N independent 3×3 systems via Cramer's rule.

    H   : (N, 3, 3)
    rhs : (N, 3)
    Returns x : (N, 3)
    """
    h00, h01, h02 = H[:, 0, 0], H[:, 0, 1], H[:, 0, 2]
    h10, h11, h12 = H[:, 1, 0], H[:, 1, 1], H[:, 1, 2]
    h20, h21, h22 = H[:, 2, 0], H[:, 2, 1], H[:, 2, 2]
    b0, b1, b2 = rhs[:, 0], rhs[:, 1], rhs[:, 2]

    c00 = h11 * h22 - h12 * h21
    c01 = h10 * h22 - h12 * h20
    c02 = h10 * h21 - h11 * h20

    det = h00 * c00 - h01 * c01 + h02 * c02
    inv_det = 1.0 / np.where(np.abs(det) > 1e-30, det, 1e-30)

    x0 = (b0 * c00 - h01 * (b1 * h22 - h12 * b2) + h02 * (b1 * h21 - h11 * b2)) * inv_det
    x1 = (h00 * (b1 * h22 - h12 * b2) - b0 * c01 + h02 * (h10 * b2 - b1 * h20)) * inv_det
    x2 = (h00 * (h11 * b2 - b1 * h21) - h01 * (h10 * b2 - b1 * h20) + b0 * c02) * inv_det

    return np.column_stack([x0, x1, x2])


class VBDSolver(BaseSolver):

    @property
    def name(self) -> str:
        return "VBD (Vertex Block Descent)"

    # ------------------------------------------------------------------
    def step(self, sim) -> None:
        dt_sub = DT / NUM_SUBSTEPS
        iters = max(sim.iterations // NUM_SUBSTEPS, 3)
        for _ in range(NUM_SUBSTEPS):
            self._substep(sim, dt_sub, iters)

    # ------------------------------------------------------------------
    def _substep(self, sim, dt, iters) -> None:
        h2 = dt * dt
        n = sim.n
        mask = sim.inv_mass > 0

        pre_vel_y = sim.vel[:, 1].copy()
        was_above = sim.pos[:, 1] > FLOOR_Y + 1.0

        w = 10.0 + sim.stiffness * 990.0
        omega = 0.65

        mass_h2 = np.full(n, 1.0 / h2)
        mass_h2[sim.pin_mask] = 1e6

        # Inertial position
        sim.vel[mask] += GRAVITY * dt
        sim.vel *= sim.damping
        s_n = sim.pos + dt * sim.vel
        s_n[sim.pin_mask] = sim.pinned_pos[sim.pin_mask]

        self._clamp_floor(sim, s_n)
        self._clamp_ball(sim, s_n)

        q = s_n.copy()
        if sim.grabbed >= 0:
            q[sim.grabbed] = sim.mouse

        ci = sim.c_i
        cj = sim.c_j

        for _ in range(iters):
            diff = q[ci] - q[cj]
            dist = np.linalg.norm(diff, axis=1)
            dist_safe = np.maximum(dist, 1e-7)
            n_dir = diff / dist_safe[:, np.newaxis]
            C = dist - sim.c_rest

            # Per-vertex gradient
            g = mass_h2[:, np.newaxis] * (q - s_n)
            g_c = (w * C)[:, np.newaxis] * n_dir
            np.add.at(g, ci,  g_c)
            np.add.at(g, cj, -g_c)

            # Per-vertex Hessian
            H = np.zeros((n, 3, 3))
            H[:, 0, 0] = mass_h2
            H[:, 1, 1] = mass_h2
            H[:, 2, 2] = mass_h2

            nn = n_dir[:, :, np.newaxis] * n_dir[:, np.newaxis, :]
            np.add.at(H, ci, w * nn)
            np.add.at(H, cj, w * nn)

            # Solve H·Δx = −g  (Cramer's rule, no LAPACK)
            delta = _solve3x3_batch(H, -g)

            q[mask] += omega * delta[mask]

            # Pin overrides
            q[sim.pin_mask] = sim.pinned_pos[sim.pin_mask]
            if sim.grabbed >= 0:
                q[sim.grabbed] = sim.mouse

            # Collision
            self._clamp_floor(sim, q)
            self._clamp_ball(sim, q)
            if sim.self_collision:
                self._self_collision(sim, q)
            if sim.grabbed >= 0:
                q[sim.grabbed] = sim.mouse

        self._collision_pass(sim, q)

        sim.vel[mask] = (q[mask] - sim.pos[mask]) / dt
        sim.pos[mask] = q[mask]
        self._post_step_bounce(sim, pre_vel_y, was_above)
