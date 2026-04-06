"""
MAP-Elites Grid: Maintains diversity of problem generation programs.

Axes:
  - Axis 1 (H bins): Entropy bins for difficulty calibration
  - Axis 2 (Embedding cluster): Diversity via problem embedding clusters

Each cell holds the champion program (highest R_Q) for that niche.
"""

import json
import os
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from .program import ProblemProgram


@dataclass
class NicheInfo:
    """Information about a single niche in the grid."""
    h_bin: int
    div_bin: int
    champion: Optional[ProblemProgram] = None
    champion_rq: float = -1.0
    update_count: int = 0
    selection_count: int = 0   # UCB: 이 셀이 부모로 선택된 횟수
    history: list = field(default_factory=list)


class MAPElitesGrid:
    """
    MAP-Elites grid for problem generation programs.
    
    Grid structure: n_h_bins × n_div_bins
    Each cell maintains the champion program with highest R_Q.
    """

    def __init__(
        self,
        n_h_bins: int = 6,
        n_div_bins: int = 6,
        h_range: tuple = (0.0, 5.0),
        embedding_model_name: str = "all-MiniLM-L6-v2",
        ucb_c: float = 1.0,
    ):
        self.n_h_bins = n_h_bins
        self.n_div_bins = n_div_bins
        self.h_range = h_range
        self.embedding_model_name = embedding_model_name
        self.ucb_c = ucb_c   # UCB exploration 계수 (높을수록 탐험 중시)

        # Initialize grid
        self.grid: dict[tuple[int, int], NicheInfo] = {}
        for i in range(n_h_bins):
            for j in range(n_div_bins):
                self.grid[(i, j)] = NicheInfo(h_bin=i, div_bin=j)

        # Embedding model (lazy loaded)
        self._embed_model = None

        # PCA components for diversity axis (fitted on first batch)
        self._pca = None
        self._div_bins_edges = None

        # Stats
        self.total_insertions = 0
        self.total_replacements = 0
        self.total_selections = 0   # UCB: 전체 부모 선택 횟수

    @property
    def embed_model(self):
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embed_model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                print("[MAP-Elites] sentence-transformers not available. "
                      "Using hash-based diversity instead.")
        return self._embed_model

    def h_to_bin(self, h_value: float) -> int:
        """Map entropy value to H bin index."""
        h_min, h_max = self.h_range
        h_clipped = max(h_min, min(h_max, h_value))
        bin_width = (h_max - h_min) / self.n_h_bins
        bin_idx = int((h_clipped - h_min) / bin_width)
        return min(bin_idx, self.n_h_bins - 1)

    def problem_to_div_bin(self, problem_text: str) -> int:
        """
        Map problem text to diversity bin index using embedding.
        Falls back to hash-based assignment if embedding model unavailable.
        """
        if self.embed_model is not None:
            embedding = self.embed_model.encode([problem_text])[0]

            if self._pca is not None:
                # Project to 1D using first PCA component
                projected = np.dot(embedding, self._pca)
                # Quantize to bins
                if self._div_bins_edges is not None:
                    bin_idx = np.digitize(projected, self._div_bins_edges) - 1
                    return max(0, min(bin_idx, self.n_div_bins - 1))

            # Before PCA is fitted, use hash of embedding
            hash_val = hash(embedding.tobytes()) % self.n_div_bins
            return hash_val
        else:
            # Fallback: hash-based
            return hash(problem_text) % self.n_div_bins

    def fit_diversity_axis(self, problems: list[str]):
        """
        Fit PCA and bin edges for the diversity axis using a batch of problems.
        Call this once with the initial seed problems.
        """
        if self.embed_model is None:
            return

        embeddings = self.embed_model.encode(problems)

        # Fit PCA (just first component for 1D)
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        self._pca = Vt[0]  # First principal component

        # Project all embeddings
        projected = centered @ self._pca
        # Compute bin edges using quantiles
        quantiles = np.linspace(0, 100, self.n_div_bins + 1)
        self._div_bins_edges = np.percentile(projected, quantiles[1:-1])

        print(f"[MAP-Elites] Diversity axis fitted with {len(problems)} problems, "
              f"{self.n_div_bins} bins")

    def try_insert(
        self,
        program: ProblemProgram,
        h_value: float,
        problem_text: str,
        rq_score: float,
    ) -> bool:
        """
        Try to insert a program into the grid.
        
        Returns True if inserted (new niche or beat champion).
        """
        h_bin = self.h_to_bin(h_value)
        div_bin = self.problem_to_div_bin(problem_text)

        program.niche_h = h_bin
        program.niche_div = div_bin
        program.rq_score = rq_score

        niche = self.grid[(h_bin, div_bin)]

        if niche.champion is None or rq_score > niche.champion_rq:
            old_rq = niche.champion_rq if niche.champion else -1
            niche.champion = program
            niche.champion_rq = rq_score
            niche.update_count += 1
            niche.history.append({
                "program_id": program.program_id,
                "rq": rq_score,
                "generation": program.generation,
            })

            self.total_insertions += 1
            if old_rq >= 0:
                self.total_replacements += 1

            return True

        return False

    def rebin_champion(
        self,
        program: ProblemProgram,
        new_h_value: float,
        problem_text: str,
    ) -> bool:
        """
        챔피언의 H값이 갱신됐을 때 그리드 내 위치를 재조정한다.

        1. 기존 niche에서 제거
        2. 새 H값으로 bin 재계산
        3. 새 niche에 try_insert (더 높은 R_Q만 삽입)

        Returns:
            True if the program ended up in a different niche
        """
        old_h_bin = program.niche_h
        old_div_bin = program.niche_div
        new_h_bin = self.h_to_bin(new_h_value)
        new_div_bin = self.problem_to_div_bin(problem_text)

        # 위치가 바뀌지 않으면 h_score만 갱신하고 종료
        if new_h_bin == old_h_bin and new_div_bin == old_div_bin:
            program.h_score = new_h_value
            # R_Q도 새 H로 재계산
            new_rq = program.p_hat * (1.0 - program.p_hat) * new_h_value
            program.rq_score = new_rq
            program.fitness = new_rq
            niche = self.grid[(old_h_bin, old_div_bin)]
            niche.champion_rq = new_rq
            return False

        # 기존 niche에서 이 프로그램이 챔피언이면 비워준다
        old_niche = self.grid.get((old_h_bin, old_div_bin))
        if old_niche and old_niche.champion is not None:
            if old_niche.champion.program_id == program.program_id:
                old_niche.champion = None
                old_niche.champion_rq = -1.0

        # 새 H로 R_Q 재계산 후 삽입 시도
        program.h_score = new_h_value
        new_rq = program.p_hat * (1.0 - program.p_hat) * new_h_value
        program.rq_score = new_rq
        program.fitness = new_rq

        self.try_insert(
            program=program,
            h_value=new_h_value,
            problem_text=problem_text,
            rq_score=new_rq,
        )
        return True

    def sample_parent(self) -> Optional[ProblemProgram]:
        """
        UCB1 기반 부모 셀 선택.

        UCB1(i) = μ_i + C * sqrt(ln(N) / n_i)

          μ_i  : 정규화된 RQ 점수 (exploitation)
          C    : 탐험 계수 (self.ucb_c)
          N    : 전체 선택 횟수
          n_i  : 셀 i의 선택 횟수

        한 번도 선택되지 않은 셀은 무한대로 처리해 먼저 탐험.
        """
        occupied = [
            (k, niche) for k, niche in self.grid.items()
            if niche.champion is not None
        ]
        if not occupied:
            return None

        self.total_selections += 1
        N = self.total_selections

        rqs = np.array([niche.champion_rq for _, niche in occupied], dtype=float)
        counts = np.array([niche.selection_count for _, niche in occupied], dtype=float)

        # RQ를 [0, 1]로 정규화 (exploitation term)
        rq_min, rq_max = rqs.min(), rqs.max()
        norm_rqs = (rqs - rq_min) / (rq_max - rq_min) if rq_max > rq_min else np.ones(len(rqs))

        # UCB1: 미방문 셀은 inf → 반드시 먼저 선택됨
        exploration = np.where(
            counts == 0,
            np.inf,
            self.ucb_c * np.sqrt(np.log(N) / counts),
        )
        ucb_scores = norm_rqs + exploration

        idx = int(np.argmax(ucb_scores))
        _, niche = occupied[idx]
        niche.selection_count += 1
        return niche.champion

    def get_all_champions(self) -> list[ProblemProgram]:
        """Get all champion programs from the grid."""
        champions = []
        for niche in self.grid.values():
            if niche.champion is not None:
                champions.append(niche.champion)
        return champions

    def coverage(self) -> float:
        """Fraction of niches that are populated."""
        occupied = sum(1 for n in self.grid.values() if n.champion is not None)
        return occupied / len(self.grid)

    def mean_rq(self) -> float:
        """Mean R_Q across all champions."""
        rqs = [n.champion_rq for n in self.grid.values() if n.champion is not None]
        return np.mean(rqs) if rqs else 0.0

    def stats(self) -> dict:
        """Get grid statistics."""
        champions = self.get_all_champions()
        rqs = [p.rq_score for p in champions]
        return {
            "coverage": self.coverage(),
            "num_champions": len(champions),
            "total_niches": len(self.grid),
            "mean_rq": np.mean(rqs) if rqs else 0.0,
            "max_rq": max(rqs) if rqs else 0.0,
            "min_rq": min(rqs) if rqs else 0.0,
            "total_insertions": self.total_insertions,
            "total_replacements": self.total_replacements,
            "total_selections": self.total_selections,
        }

    def save(self, path: str):
        """Save grid state to directory."""
        os.makedirs(path, exist_ok=True)

        # Save metadata
        meta = {
            "n_h_bins": self.n_h_bins,
            "n_div_bins": self.n_div_bins,
            "h_range": self.h_range,
            "stats": self.stats(),
        }
        with open(os.path.join(path, "grid_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Save champions
        for (h, d), niche in self.grid.items():
            if niche.champion is not None:
                niche.champion.save(
                    os.path.join(path, f"champion_{h}_{d}.json")
                )

    def load(self, path: str):
        """Load grid state from directory."""
        for fname in os.listdir(path):
            if fname.startswith("champion_") and fname.endswith(".json"):
                parts = fname.replace("champion_", "").replace(".json", "").split("_")
                h_bin, div_bin = int(parts[0]), int(parts[1])
                program = ProblemProgram.load(os.path.join(path, fname))
                niche = self.grid.get((h_bin, div_bin))
                if niche:
                    niche.champion = program
                    niche.champion_rq = program.rq_score
