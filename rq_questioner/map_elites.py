from __future__ import annotations

import json
import os
import random as _random
import hashlib
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
    candidates: list[ProblemProgram] = field(default_factory=list)
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
        candidate_reservoir_size: int = 4,
    ):
        self.n_h_bins = n_h_bins
        self.n_div_bins = n_div_bins
        self.h_range = h_range
        self.ucb_c = ucb_c
        self.epsilon = epsilon
        self.candidate_reservoir_size = max(0, int(candidate_reservoir_size))

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
        self.total_reservoir_insertions = 0
        self.total_reservoir_selections = 0
        self.total_duplicate_rejections = 0

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

    @staticmethod
    def _normalize_signature_text(text: str) -> str:
        return " ".join(str(text or "").strip().split())

    @classmethod
    def program_behavior_signature(
        cls,
        program: ProblemProgram,
        n_seeds: int = 5,
    ) -> str | None:
        """Signature of seed-indexed behavior, independent of program_id.

        This catches near-clone programs that generate the exact same
        problem/answer sequence for the verification seeds but land in
        different MAP cells due to noisy H/D estimates.
        """
        cache_key = f"_behavior_signature_v1_{n_seeds}"
        cached = (program.metadata or {}).get(cache_key)
        if cached:
            return str(cached)

        pairs = []
        for seed in range(n_seeds):
            inst = program.execute(seed=seed, timeout=5.0)
            if inst is None:
                return None
            pairs.append((
                cls._normalize_signature_text(inst.problem),
                cls._normalize_signature_text(inst.answer),
            ))
        payload = json.dumps(pairs, ensure_ascii=False, sort_keys=True)
        signature = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        program.metadata[cache_key] = signature
        return signature

    def _find_duplicate_behavior(
        self,
        program: ProblemProgram,
    ) -> ProblemProgram | None:
        signature = self.program_behavior_signature(program)
        if not signature:
            return None

        for niche in self.grid.values():
            material = []
            if niche.champion is not None:
                material.append(niche.champion)
            material.extend(niche.candidates)
            for existing in material:
                if existing.program_id == program.program_id:
                    continue
                if self.program_behavior_signature(existing) == signature:
                    program.metadata["_behavior_signature_v1_5"] = signature
                    return existing
        return None

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

        duplicate = self._find_duplicate_behavior(program)
        if duplicate is not None:
            program.metadata["archive_status"] = "duplicate_rejected"
            program.metadata["reservoir_reason"] = "duplicate_behavior_signature"
            program.metadata["duplicate_of"] = duplicate.program_id
            niche.history.append({
                "program_id": program.program_id,
                "rq": rq_score,
                "generation": program.generation,
                "event": "duplicate_rejected",
                "duplicate_of": duplicate.program_id,
            })
            self.total_duplicate_rejections += 1
            return False

        if niche.champion is None or rq_score > niche.champion_rq:
            displaced = niche.champion
            old_rq = niche.champion_rq if niche.champion else -1
            niche.champion = program
            niche.champion_rq = rq_score
            program.metadata["archive_status"] = "champion"
            program.metadata.pop("reservoir_reason", None)
            niche.update_count += 1
            niche.history.append({
                "program_id": program.program_id,
                "rq": rq_score,
                "generation": program.generation,
                "event": "inserted" if old_rq < 0 else "replaced",
            })
            self.total_insertions += 1
            if old_rq >= 0:
                self.total_replacements += 1
                if displaced is not None and displaced.program_id != program.program_id:
                    self._add_candidate_to_reservoir(
                        displaced,
                        h_bin,
                        div_bin,
                        reason="displaced_champion",
                    )
            return True

        self._add_candidate_to_reservoir(
            program,
            h_bin,
            div_bin,
            reason="rejected_non_elite",
        )
        return False

    def _add_candidate_to_reservoir(
        self,
        program: ProblemProgram,
        h_bin: int,
        div_bin: int,
        reason: str,
    ) -> bool:
        """Keep non-champion but valid/scored programs as evolution material."""
        if self.candidate_reservoir_size <= 0:
            return False

        niche = self.grid[(h_bin, div_bin)]
        if (
            niche.champion is not None
            and niche.champion.program_id == program.program_id
        ):
            return False

        for existing in niche.candidates:
            if existing.program_id == program.program_id:
                return False

        program.niche_h = h_bin
        program.niche_div = div_bin
        program.metadata["archive_status"] = "reservoir"
        program.metadata["reservoir_reason"] = reason
        niche.candidates.append(program)
        niche.candidates.sort(
            key=lambda p: (
                float(getattr(p, "rq_score", 0.0) or 0.0),
                int(getattr(p, "generation", 0) or 0),
            ),
            reverse=True,
        )
        del niche.candidates[self.candidate_reservoir_size:]
        niche.history.append({
            "program_id": program.program_id,
            "rq": float(getattr(program, "rq_score", 0.0) or 0.0),
            "generation": program.generation,
            "event": reason,
        })
        self.total_reservoir_insertions += 1
        return True

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

    def _champion_entries(self) -> list:
        return [
            (k, niche) for k, niche in self.grid.items()
            if niche.champion is not None
        ]

    def _reservoir_entries(self) -> list:
        entries = []
        for key, niche in self.grid.items():
            for candidate in niche.candidates:
                if (
                    niche.champion is not None
                    and niche.champion.program_id == candidate.program_id
                ):
                    continue
                entries.append((key, niche, candidate))
        return entries

    def _material_entries(self) -> list:
        """All programs available as evolution parents: champions + reservoir."""
        entries = []
        for key, niche in self.grid.items():
            if niche.champion is not None:
                entries.append((key, niche, niche.champion, "champion"))
            for candidate in niche.candidates:
                if (
                    niche.champion is not None
                    and niche.champion.program_id == candidate.program_id
                ):
                    continue
                entries.append((key, niche, candidate, "reservoir"))
        return entries

    @staticmethod
    def _program_selection_count(program: ProblemProgram) -> int:
        return int((program.metadata or {}).get("_selection_count", 0) or 0)

    def _increment_parent_selection(
        self, niche: NicheInfo, program: ProblemProgram, source: str
    ) -> None:
        self.total_selections += 1
        niche.selection_count += 1
        program.metadata["_selection_count"] = (
            self._program_selection_count(program) + 1
        )
        if source == "reservoir":
            self.total_reservoir_selections += 1

    def _ucb_scores(self, occupied: list) -> np.ndarray:
        """Rank-based UCB scores for occupied niches."""
        N = self.total_selections + 1
        rqs = np.array([niche.champion_rq for _, niche in occupied], dtype=float)
        counts = np.array([niche.selection_count for _, niche in occupied], dtype=float)

        # Rank normalization (균등 분포, outlier에 강건)
        ranks = np.argsort(np.argsort(rqs)).astype(float)
        n = len(ranks)
        norm_rqs = ranks / (n - 1) if n > 1 else np.ones(n)

        exploration = np.full_like(counts, np.inf, dtype=float)
        seen = counts > 0
        exploration[seen] = self.ucb_c * np.sqrt(np.log(N) / counts[seen])
        return norm_rqs + exploration

    def _material_ucb_scores(self, entries: list) -> np.ndarray:
        """Rank-based UCB over champion and non-champion material together."""
        N = self.total_selections + 1
        rqs = np.array(
            [
                float(getattr(program, "rq_score", 0.0) or 0.0)
                for _, _, program, _ in entries
            ],
            dtype=float,
        )
        counts = np.array(
            [self._program_selection_count(program) for _, _, program, _ in entries],
            dtype=float,
        )

        ranks = np.argsort(np.argsort(rqs)).astype(float)
        n = len(ranks)
        norm_rqs = ranks / (n - 1) if n > 1 else np.ones(n)

        exploration = np.full_like(counts, np.inf, dtype=float)
        seen = counts > 0
        exploration[seen] = self.ucb_c * np.sqrt(np.log(N) / counts[seen])
        return norm_rqs + exploration

    def _sample_material_parent(
        self, exclude_program_ids: set[str] | None = None,
    ) -> Optional[ProblemProgram]:
        entries = self._material_entries()
        if exclude_program_ids:
            entries = [
                entry for entry in entries
                if entry[2].program_id not in exclude_program_ids
            ]
        if not entries:
            return None

        if _random.random() < self.epsilon:
            _, niche, program, source = _random.choice(entries)
        else:
            ucb_scores = self._material_ucb_scores(entries)
            idx = int(np.argmax(ucb_scores))
            _, niche, program, source = entries[idx]

        self._increment_parent_selection(niche, program, source)
        return program

    def sample_parent(self) -> Optional[ProblemProgram]:
        """
        Parent selection over champion and non-champion material together.

        ε 확률: all material에서 uniform random
        (1-ε) 확률: R_Q rank + UCB exploration
        """
        return self._sample_material_parent()

    def sample_two_parents(self) -> tuple[Optional[ProblemProgram], Optional[ProblemProgram]]:
        """Crossover용 부모 2개 선택 (champion + reservoir material)."""
        material: dict[str, ProblemProgram] = {}
        for _, niche in self._champion_entries():
            material[niche.champion.program_id] = niche.champion
        for _, _, candidate in self._reservoir_entries():
            material[candidate.program_id] = candidate

        if len(material) < 2:
            return None, None

        first = self._sample_material_parent()
        if first is None:
            return None, None
        second = self._sample_material_parent({first.program_id})
        if second is None:
            first, second = _random.sample(list(material.values()), 2)
        return first, second

    # ------------------------------------------------------------------
    # Stats & IO
    # ------------------------------------------------------------------

    def get_all_champions(self) -> list[ProblemProgram]:
        return [n.champion for n in self.grid.values() if n.champion is not None]

    def get_all_reservoir_candidates(self) -> list[ProblemProgram]:
        return [p for n in self.grid.values() for p in n.candidates]

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
        reservoir_candidates = self.get_all_reservoir_candidates()
        return {
            "coverage": self.coverage(),
            "num_champions": len(champions),
            "num_reservoir_candidates": len(reservoir_candidates),
            "num_reservoir_cells": sum(1 for n in self.grid.values() if n.candidates),
            "hard_champions": self.count_hard_champions(min_h_bin=2),
            "total_niches": len(self.grid),
            "mean_rq": float(np.mean(rqs)) if rqs else 0.0,
            "max_rq": float(max(rqs)) if rqs else 0.0,
            "min_rq": float(min(rqs)) if rqs else 0.0,
            "total_insertions": self.total_insertions,
            "total_replacements": self.total_replacements,
            "total_selections": self.total_selections,
            "total_reservoir_insertions": self.total_reservoir_insertions,
            "total_reservoir_selections": self.total_reservoir_selections,
            "total_duplicate_rejections": self.total_duplicate_rejections,
            "candidate_reservoir_size": self.candidate_reservoir_size,
        }

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        meta = {
            "n_h_bins": self.n_h_bins,
            "n_div_bins": self.n_div_bins,
            "h_range": self.h_range,
            "candidate_reservoir_size": self.candidate_reservoir_size,
            "total_duplicate_rejections": self.total_duplicate_rejections,
            "stats": self.stats(),
        }
        with open(os.path.join(path, "grid_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        for (h, d), niche in self.grid.items():
            if niche.champion is not None:
                niche.champion.save(os.path.join(path, f"champion_{h}_{d}.json"))
            for idx, candidate in enumerate(niche.candidates):
                candidate.save(os.path.join(path, f"candidate_{h}_{d}_{idx}.json"))

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
            elif fname.startswith("candidate_") and fname.endswith(".json"):
                parts = fname.replace("candidate_", "").replace(".json", "").split("_")
                h_bin, div_bin = int(parts[0]), int(parts[1])
                program = ProblemProgram.load(os.path.join(path, fname))
                niche = self.grid.get((h_bin, div_bin))
                if niche:
                    niche.candidates.append(program)
                    niche.candidates.sort(
                        key=lambda p: (
                            float(getattr(p, "rq_score", 0.0) or 0.0),
                            int(getattr(p, "generation", 0) or 0),
                        ),
                        reverse=True,
                    )
                    del niche.candidates[self.candidate_reservoir_size:]
