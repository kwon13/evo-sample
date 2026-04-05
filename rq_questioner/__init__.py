from .program import ProblemProgram
from .map_elites import MAPElitesGrid
from .mutator import ProgramMutator
from .verifier import verify_problem
from .rq_score import compute_rq
from .pipeline import EvolutionaryPipeline

# GPU-dependent modules: import explicitly when needed
# from .entropy import measure_entropy
