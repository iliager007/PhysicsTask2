from .base import BaseSolver


class VBDSolver(BaseSolver):

    @property
    def name(self) -> str:
        return "VBD (Vertex Block Descent)"

    @property
    def implemented(self) -> bool:
        return False

    def step(self, sim) -> None:
        pass  # TODO: implement
