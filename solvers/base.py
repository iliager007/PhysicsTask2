"""Abstract base class for all cloth physics solvers."""
from abc import ABC, abstractmethod


class BaseSolver(ABC):
    """Contract that every solver must satisfy.

    A solver receives the live ``Simulation`` instance (which carries all
    particle state and parameters) and advances it by one timestep.
    Rendering is always handled by ``Simulation``; solvers only mutate state.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable label shown in the UI and HUD."""
        ...

    @property
    def implemented(self) -> bool:
        """Return ``False`` for stub solvers not yet coded."""
        return True

    @abstractmethod
    def step(self, sim) -> None:
        """Advance the simulation by one fixed timestep.

        Parameters
        ----------
        sim:
            The live ``Simulation`` instance.  Solvers read solver
            parameters (``sim.iterations``, ``sim.stiffness``, etc.) and
            read/write particle arrays (``sim.pos``, ``sim.vel``,
            ``sim.predicted``).
        """
        ...
