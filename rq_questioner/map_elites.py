import json
import os
import random as _random
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from .program import ProblemProgram


@dataclass
class NicheInfo:
    h_bin: int
    div_bin: int
    champion: Optional[ProblemProgram] = None
    champion_rq: float = -1.0
    update_count: int = 0
    selection_count: int = 0
    history: list = field(default_factory=list)


class MAPElitesGrid:
    """
    MAP-Elites grid: H bins (entropy/difficulty) x D bins (embedding diversity).

    D-axis uses sentence embedding + PCA (연구 제안서 3.3절):
      - all-MiniLM-L6-v2 로 문제 텍스트를 embed
      - PC1 값을 균등 분할하여 D bin 결정
      - fit_diversity_axis()로 seed 문제 기반 초기 fitting

    Parent selection: ε-greedy + rank-based UCB (Monte Carlo Elites 방식):
      - ε 확률로 uniform random (exploration)
      - (1-ε) 확률로 rank-based UCB (exploitation)
    """

    def __init__(
        self,
        n_h_bins: int = 6,
        n_div_bins: int = 10,
        h_range: tuple = (0.0, 5.0),
        ucb_c: float = 1.0,
        epsilon: float = 0.3,
        seed_ids: list[str] | None = None,
    ):
        self.n_h_bins = n_h_bins
        self.n_div_bins = n_div_bins
        self.h_range = h_range
        self.ucb_c = ucb_c
        self.epsilon = epsilon

        # Embedding model (lazy loaded)
        self._embedder = None
        self._pca_mean = None
        self._pca_component = None  # PC1 방향 벡터
        self._pca_min = None
        self._pca_max = None
        self._diversity_fitted = False

        # seed_ids: 하위 호환용으로 유지하되 D bin 결정에는 사용하지 않음
        # (fit_diversity_axis 전까지는 hash fallback 사용)

        self.grid: dict[tuple[int, int], NicheInfo] = {}
        for i in range(n_h_bins):
            for j in range(n_div_bins):
                self.grid[(i, j)] = NicheInfo(h_bin=i, div_bin=j)

        self.total_insertions = 0
        self.total_replacements = 0
        self.total_selections = 0

    # ------------------------------------------------------------------
    # Embedding & Diversity Axis
    # ------------------------------------------------------------------

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def fit_diversity_axis(self, problem_texts: list[str]):
        """
        Seed 문제 텍스트로 PCA diversity axis를 fitting.
        PC1 방향을 구하고, 그 projection의 min/max로 D bin 경계를 설정.
        """
        if not problem_texts:
            return

        embedder = self._get_embedder()
        embeddings = embedder.encode(problem_texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)

        # PCA: PC1만 필요
        self._pca_mean = embeddings.mean(axis=0)
        centered = embeddings - self._pca_mean

        # SVD로 PC1
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        self._pca_component = Vt[0]  # shape: (dim,)

        # PC1 projection의 범위
        projections = centered @ self._pca_component
        self._pca_min = float(projections.min())
        self._pca_max = float(projections.max())

        # 약간의 여유 (새 문제가 범위를 벗어나도 clamp됨)
        margin = (self._pca_max - self._pca_min) * 0.1
        self._pca_min -= margin
        self._pca_max += margin

        self._diversity_fitted = True

    def problem_to_div_bin(self, problem_text: str) -> int:
        """
        문제 텍스트를 embed → PC1 projection → D bin.
        fit_diversity_axis()가 호출되지 않은 경우 hash fallback.
        """
        if not self._diversity_fitted or not problem_text:
            return hash(problem_text) % self.n_div_bins

        embedder = self._get_embedder()
        emb = embedder.encode([problem_text], show_progress_bar=False)[0]
        centered = emb - self._pca_mean
        proj = float(centered @ self._pca_component)

        # Clamp to fitted range
        proj = max(self._pca_min, min(self._pca_max, proj))
        bin_width = (self._pca_max - self._pca_min) / self.n_div_bins
        bin_idx = int((proj - self._pca_min) / bin_width)
        return min(bin_idx, self.n_div_bins - 1)

    # ------------------------------------------------------------------
    # H-axis
    # ------------------------------------------------------------------

    def h_to_bin(self, h_value: float) -> int:
        h_min, h_max = self.h_range
        h_clipped = max(h_min, min(h_max, h_value))
        bin_width = (h_max - h_min) / self.n_h_bins
        bin_idx = int((h_clipped - h_min) / bin_width)
        return min(bin_idx, self.n_h_bins - 1)

    # ------------------------------------------------------------------
    # Grid Operations
    # ------------------------------------------------------------------

    def register_seed(self, seed_id: str) -> int:
        """하위 호환용. embedding 기반에서는 no-op."""
        return 0

    def try_insert(
        self,
        program: ProblemProgram,
        h_value: float,
        problem_text: str = "",
        rq_score: float = 0.0,
    ) -> bool:
        """
        문제 텍스트 embedding 기반으로 D bin 결정 후 삽입 시도.
        빈 niche이거나 기존 champion보다 R_Q가 높으면 삽입.
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

    def evict_champion(self, program: ProblemProgram) -> bool:
        """
        Champion re-evaluation 결과 p_hat 이 극단(0 또는 1 근처) 이 된 경우
        해당 niche 를 비워 다음 mutation 라운드에 재탐색되도록 한다.

        Returns:
            True  — 실제로 eviction 발생
            False — 주어진 program 이 현재 champion 이 아니거나 niche 가 비어있음
        """
        niche = self.grid.get((program.niche_h, program.niche_div))
        if niche is None or niche.champion is None:
            return False
        if niche.champion.program_id != program.program_id:
            return False

        niche.champion = None
        niche.champion_rq = -1.0
        niche.history.append({
            "program_id": program.program_id,
            "event": "evicted",
            "generation": program.generation,
        })
        return True

    def rebin_champion(
        self,
        program: ProblemProgram,
        new_h_value: float,
        problem_text: str,
    ) -> bool:
        old_h_bin = program.niche_h
        old_div_bin = program.niche_div
        new_h_bin = self.h_to_bin(new_h_value)
        new_div_bin = self.problem_to_div_bin(problem_text)

        if new_h_bin == old_h_bin and new_div_bin == old_div_bin:
            program.h_score = new_h_value
            new_rq = program.p_hat * (1.0 - program.p_hat) * new_h_value
            program.rq_score = new_rq
            program.fitness = new_rq
            niche = self.grid[(old_h_bin, old_div_bin)]
            niche.champion_rq = new_rq
            return False

        old_niche = self.grid.get((old_h_bin, old_div_bin))
        if old_niche and old_niche.champion is not None:
            if old_niche.champion.program_id == program.program_id:
                old_niche.champion = None
                old_niche.champion_rq = -1.0

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

    # ------------------------------------------------------------------
    # Parent Selection: ε-greedy + rank-based UCB
    # ------------------------------------------------------------------

    def _ucb_scores(self, occupied: list) -> np.ndarray:
        """Rank-based UCB scores for occupied niches."""
        N = self.total_selections + 1
        rqs = np.array([niche.champion_rq for _, niche in occupied], dtype=float)
        counts = np.array([niche.selection_count for _, niche in occupied], dtype=float)

        # Rank normalization (균등 분포, outlier에 강건)
        ranks = np.argsort(np.argsort(rqs)).astype(float)
        n = len(ranks)
        norm_rqs = ranks / (n - 1) if n > 1 else np.ones(n)

        exploration = np.where(
            counts == 0,
            np.inf,
            self.ucb_c * np.sqrt(np.log(N) / counts),
        )
        return norm_rqs + exploration

    def sample_parent(self) -> Optional[ProblemProgram]:
        """
        ε-greedy + rank-based UCB 부모 선택.

        ε 확률: uniform random (모든 occupied 셀에서 균등 선택)
        (1-ε) 확률: rank-based UCB1 (exploitation + exploration)
        """
        occupied = [
            (k, niche) for k, niche in self.grid.items()
            if niche.champion is not None
        ]
        if not occupied:
            return None

        self.total_selections += 1

        if _random.random() < self.epsilon:
            _, niche = _random.choice(occupied)
            niche.selection_count += 1
            return niche.champion

        ucb_scores = self._ucb_scores(occupied)
        idx = int(np.argmax(ucb_scores))
        _, niche = occupied[idx]
        niche.selection_count += 1
        return niche.champion

    def sample_two_parents(self) -> tuple[Optional[ProblemProgram], Optional[ProblemProgram]]:
        """Crossover용 부모 2개 선택 (서로 다른 셀)."""
        occupied = [
            (k, niche) for k, niche in self.grid.items()
            if niche.champion is not None
        ]
        if len(occupied) < 2:
            return None, None

        self.total_selections += 2

        if _random.random() < self.epsilon:
            pair = _random.sample(occupied, 2)
            pair[0][1].selection_count += 1
            pair[1][1].selection_count += 1
            return pair[0][1].champion, pair[1][1].champion

        ucb_scores = self._ucb_scores(occupied)

        idx1 = int(np.argmax(ucb_scores))
        _, niche1 = occupied[idx1]
        niche1.selection_count += 1

        ucb_scores[idx1] = -np.inf
        idx2 = int(np.argmax(ucb_scores))
        _, niche2 = occupied[idx2]
        niche2.selection_count += 1

        return niche1.champion, niche2.champion

    # ------------------------------------------------------------------
    # Stats & IO
    # ------------------------------------------------------------------

    def get_all_champions(self) -> list[ProblemProgram]:
        return [n.champion for n in self.grid.values() if n.champion is not None]

    def coverage(self) -> float:
        occupied = sum(1 for n in self.grid.values() if n.champion is not None)
        return occupied / len(self.grid)

    def mean_rq(self) -> float:
        rqs = [n.champion_rq for n in self.grid.values() if n.champion is not None]
        return float(np.mean(rqs)) if rqs else 0.0

    def count_hard_champions(self, min_h_bin: int = 2) -> int:
        return sum(
            1 for (h, _), n in self.grid.items()
            if h >= min_h_bin and n.champion is not None
        )

    def stats(self) -> dict:
        champions = self.get_all_champions()
        rqs = [p.rq_score for p in champions]
        return {
            "coverage": self.coverage(),
            "num_champions": len(champions),
            "hard_champions": self.count_hard_champions(min_h_bin=2),
            "total_niches": len(self.grid),
            "mean_rq": float(np.mean(rqs)) if rqs else 0.0,
            "max_rq": float(max(rqs)) if rqs else 0.0,
            "min_rq": float(min(rqs)) if rqs else 0.0,
            "total_insertions": self.total_insertions,
            "total_replacements": self.total_replacements,
            "total_selections": self.total_selections,
        }

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        meta = {
            "n_h_bins": self.n_h_bins,
            "n_div_bins": self.n_div_bins,
            "h_range": self.h_range,
            "stats": self.stats(),
        }
        with open(os.path.join(path, "grid_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        for (h, d), niche in self.grid.items():
            if niche.champion is not None:
                niche.champion.save(os.path.join(path, f"champion_{h}_{d}.json"))

    def load(self, path: str):
        for fname in os.listdir(path):
            if fname.startswith("champion_") and fname.endswith(".json"):
                parts = fname.replace("champion_", "").replace(".json", "").split("_")
                h_bin, div_bin = int(parts[0]), int(parts[1])
                program = ProblemProgram.load(os.path.join(path, fname))
                niche = self.grid.get((h_bin, div_bin))
                if niche:
                    niche.champion = program
                    niche.champion_rq = program.rq_score
