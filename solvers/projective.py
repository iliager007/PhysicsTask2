from .base import BaseSolver


class ProjectiveDynamicsSolver(BaseSolver):

    @property
    def name(self) -> str:
        return "Projective Dynamics"

    @property
    def implemented(self) -> bool:
        return False

    def step(self, sim) -> None:
        pass  # TODO: implement
