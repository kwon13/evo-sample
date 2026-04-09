from .program import ProblemProgram
from .map_elites import MAPElitesGrid
from .verifier import verify_problem
from .rq_score import compute_rq
from .pipeline import EvolutionaryPipeline


def __getattr__(name):
    if name == "MapElitesDynamicDataset":
        from .verl_dataset import MapElitesDynamicDataset
        return MapElitesDynamicDataset
    if name == "RQEvolveTrainer":
        from .verl_trainer import RQEvolveTrainer
        return RQEvolveTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
