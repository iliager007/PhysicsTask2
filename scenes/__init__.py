from .hanging import (
    SinglePointScene,
    TwoCornersScene,
    ThreePointsScene,
    FullTopEdgeScene,
)
from .floor_collision import (
    FreeFallScene,
    DropOntoBallScene,
    DenseCrumpleScene,
)
from .overconstrained import (
    TwoDistantCornersScene,
    FourCornersSpreadScene,
    CompressedEdgesScene,
)

SCENES: list = [
    # Category 1 — Hanging
    SinglePointScene(),
    TwoCornersScene(),
    ThreePointsScene(),
    FullTopEdgeScene(),
    # Category 2 — Floor & ball
    FreeFallScene(),
    DropOntoBallScene(),
    DenseCrumpleScene(),
    # Category 3 — Overconstrained
    TwoDistantCornersScene(),
    FourCornersSpreadScene(),
    CompressedEdgesScene(),
]
