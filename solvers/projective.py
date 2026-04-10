"""Projective Dynamics solver (Bouaziz et al. 2014, SIGGRAPH).

The system matrix  A = M/h² + Σ wc·Lc  depends only on topology, mass and
stiffness.  It is pre-factored (inverted) once and reused every frame.

Each iteration alternates:
  * **Local step** — project each distance constraint independently.
  * **Global step** — solve A·q = b  via the cached A⁻¹.

The frame is divided into several sub-steps so that per-step velocities
stay small enough for point-based ball collision to work reliably.
"""
import numpy as np

from .base import BaseSolver
from constants import DT, GRAVITY, FLOOR_Y

NUM_SUBSTEPS = 3


class ProjectiveDynamicsSolver(BaseSolver):

    def __init__(self):
        self._A_inv: np.ndarray | None = None
        self._w: float = 0.0
        self._cache_key: tuple = ()

    @property
    def name(self) -> str:
        return "Projective Dynamics"

    # ------------------------------------------------------------------
    # System matrix
    # ------------------------------------------------------------------
    def _build_system(self, sim, h2: float) -> None:
        n = sim.n
        w = 10.0 + sim.stiffness * 990.0

        A = np.zeros((n, n))
        np.fill_diagonal(A, 1.0 / h2)

        ci = sim.c_i.astype(int)
        cj = sim.c_j.astype(int)
        np.add.at(A, (ci, ci),  w)
        np.add.at(A, (cj, cj),  w)
        np.add.at(A, (ci, cj), -w)
        np.add.at(A, (cj, ci), -w)

        pinned = np.where(sim.pin_mask)[0]
        A[pinned, :] = 0.0
        A[pinned, pinned] = 1.0

        self._A_inv = np.linalg.inv(A)
        self._w = w

    def _ensure_system(self, sim, h2: float) -> None:
        key = (sim.n, sim.nc, round(sim.stiffness, 3), round(h2, 12))
        if key != self._cache_key:
            self._build_system(sim, h2)
            self._cache_key = key

    # ------------------------------------------------------------------
    # Public entry — splits the frame into sub-steps
    # ------------------------------------------------------------------
    def step(self, sim) -> None:
        dt_sub = DT / NUM_SUBSTEPS
        h2 = dt_sub * dt_sub
        self._ensure_system(sim, h2)
        iters = max(sim.iterations // NUM_SUBSTEPS, 3)

        for _ in range(NUM_SUBSTEPS):
            self._substep(sim, dt_sub, h2, iters)

    # ------------------------------------------------------------------
    # Single sub-step
    # ------------------------------------------------------------------
    def _substep(self, sim, dt, h2, iters) -> None:
        mask = sim.inv_mass > 0
        w = self._w
        pinned = np.where(sim.pin_mask)[0]

        pre_vel_y = sim.vel[:, 1].copy()
        was_above = sim.pos[:, 1] > FLOOR_Y + 1.0

        # Inertial position
        sim.vel[mask] += GRAVITY * dt
        sim.vel *= sim.damping
        s_n = sim.pos + dt * sim.vel
        s_n[pinned] = sim.pinned_pos[pinned]

        # Clamp s_n out of collisions
        self._clamp_floor(sim, s_n)
        self._clamp_ball(sim, s_n)

        q = s_n.copy()
        if sim.grabbed >= 0:
            q[sim.grabbed] = sim.mouse

        rhs_inertia = (1.0 / h2) * s_n

        for _ in range(iters):
            # --- Local step ---
            diff = q[sim.c_i] - q[sim.c_j]
            dist = np.linalg.norm(diff, axis=1)
            dist_safe = np.maximum(dist, 1e-7)
            p = sim.c_rest[:, np.newaxis] * (diff / dist_safe[:, np.newaxis])

            # --- Build RHS ---
            b = rhs_inertia.copy()
            np.add.at(b, sim.c_i,  w * p)
            np.add.at(b, sim.c_j, -w * p)
            b[pinned] = sim.pinned_pos[pinned]
            if sim.grabbed >= 0:
                b[sim.grabbed] = sim.mouse

            # --- Global step ---
            q = self._A_inv @ b

            # --- Collision ---
            self._clamp_floor(sim, q)
            self._clamp_ball(sim, q)
            if sim.grabbed >= 0:
                q[sim.grabbed] = sim.mouse

        # Final collision pass (includes self-collision)
        self._collision_pass(sim, q)
        q[pinned] = sim.pinned_pos[pinned]
        if sim.grabbed >= 0:
            q[sim.grabbed] = sim.mouse

        # Velocity & position update
        sim.vel[mask] = (q[mask] - sim.pos[mask]) / dt
        sim.pos[mask] = q[mask]
        self._post_step_bounce(sim, pre_vel_y, was_above)
