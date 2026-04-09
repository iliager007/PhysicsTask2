"""Abstract base class for all simulation scenes."""
from abc import ABC, abstractmethod


class BaseScene(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Short display name for UI dropdown and HUD."""

    @property
    @abstractmethod
    def category(self) -> str:
        """``<optgroup>`` label in the scene selector."""

    @property
    def default_self_collision(self) -> bool:
        return False

    @abstractmethod
    def setup(self, sim) -> None:
        """Initialise *sim* for this scene.

        Implementations should:

        1. Call ``sim._build_cloth(rows, cols, spacing, center_3d,
           pin_mode, plane)``  where *center_3d* is an ``(x, y, z)``
           tuple and *plane* is ``"xy"`` (vertical) or ``"xz"``
           (horizontal).
        2. Set ``sim.ball_active``, ``sim.ball_pos``, ``sim.ball_radius``.
        3. Set ``sim.camera.target``, ``sim.camera.distance`` (optional).
        """
