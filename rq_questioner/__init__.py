from .program import ProblemProgram
from .map_elites import MAPElitesGrid
from .verifier import verify_problem
from .rq_score import compute_rq
from .pipeline import EvolutionaryPipeline
from .verl_dataset import MapElitesDynamicDataset

try:
    from .verl_trainer import RQEvolveTrainer
except Exception:
    pass  # verl not installed or incompatible; RQEvolveTrainer unavailable
