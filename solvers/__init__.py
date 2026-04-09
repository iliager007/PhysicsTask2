from .xpbd import XPBDSolver
from .projective import ProjectiveDynamicsSolver
from .vbd import VBDSolver

SOLVERS: list = [
    XPBDSolver(),
    ProjectiveDynamicsSolver(),
    VBDSolver(),
]
