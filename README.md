# R_Q Evolutionary Problem Generation

R_Q 기반 진화적 문제 생성 프레임워크: KL-Regularized RL 하에서 유도된 목표 함수 R_Q = p(1-p) · H̄ 를 fitness로 사용하여, 역방향 구성(Inverse Construction) 기반 문제 생성 프로그램을 MAP-Elites로 진화시킵니다.

## 핵심 아이디어

1. **문제가 아니라 문제 공장을 진화시킨다**: 개별 문제 텍스트가 아닌, 문제 생성 프로그램(Python 함수)을 진화 단위로 사용
2. **역방향 구성**: 정답을 먼저 선택하고, 정답으로부터 문제를 수학적으로 구성 → 정답 신뢰도 보장
3. **R_Q = p(1-p) · H̄**: Learnability(p(1-p))와 gradient richness(H̄)를 동시에 포착하는 이론적으로 정당화된 fitness
4. **MAP-Elites**: H 구간 × embedding 클러스터 grid로 다양성을 구조적으로 보장

## 프로젝트 구조

```
rq_evolve/
├── run.py                     # 메인 진입점
├── requirements.txt
├── configs/
│   └── grpo_config.yaml       # veRL GRPO 설정
├── rq_questioner/             # 핵심 모듈
│   ├── __init__.py
│   ├── program.py             # ProblemProgram (진화 단위)
│   ├── verifier.py            # 역방향 구성 검증기
│   ├── entropy.py             # H 측정 (Solver forward pass)
│   ├── rq_score.py            # R_Q 계산
│   ├── map_elites.py          # MAP-Elites grid
│   ├── mutator.py             # LLM 기반 프로그램 변이
│   └── pipeline.py            # 전체 진화 파이프라인
├── seed_programs/             # 초기 시드 프로그램
│   ├── 01_polynomial.py
│   ├── 02_modular.py
│   ├── 03_linear_system.py
│   └── 04_combinatorics.py
├── data/
│   └── prepare_limr.py        # LIMR 데이터셋 준비
└── scripts/
    ├── run_full.sh             # 전체 파이프라인 실행
    └── test_local.py           # GPU 없이 로컬 검증
```

## 빠른 시작

### 1. 설치

```bash
pip install -r requirements.txt
```

### 2. 로컬 검증 (GPU 불필요)

```bash
python scripts/test_local.py
```

### 3. 시드 프로그램 준비

```bash
python run.py prepare --output_dir ./seed_programs --max_limr 500
```

### 4. 진화 실행

```bash
# Questioner 진화만 (디버깅용)
python run.py evolve \
    --seed_dir ./seed_programs \
    --output_dir ./rq_output \
    --num_epochs 3 \
    --num_generations 50

# 전체 파이프라인 (진화 + Solver GRPO 학습)
bash scripts/run_full.sh
```

## 알고리즘 (제안서 4.1절)

```
Outer Loop (Solver-Questioner co-evolution):
  for epoch in 1, 2, ...:

    Inner Loop (Questioner evolution):
      for gen in 1, ..., 100:
        1. MAP-Elites grid에서 parent 프로그램 샘플링
        2. LLM mutator가 candidate 프로그램 생성 (diff/rewrite)
        3. 각 candidate 프로그램이 문제 인스턴스 생성 (k=3~5개)
        4. verify(x', a') → 실패 시 폐기
        5. H pre-filter: 1회 forward pass → H 낮으면 폐기
        6. 통과한 문제: G=8~16 rollouts → p̂, R̂_Q 계산
        7. MAP-Elites grid 갱신 (새 niche 또는 champion 교체)

    Solver Training:
      8. 각 niche에서 최적 문제 프로그램 선택
      9. 새 seed로 문제 인스턴스 생성 → veRL GRPO 학습
     10. Solver 개선 → p 변화 → R_Q landscape 이동
```

## TODO

- [ ] `_measure_h`: 실제 Solver 모델을 사용한 entropy 측정 (현재 heuristic proxy)
- [ ] `_run_rollouts`: 실제 Solver rollout 구현 (현재 시뮬레이션)
- [ ] veRL reward function 커스터마이징 (math answer matching)
- [ ] Multi-GPU 분산 진화
- [ ] Mutator fine-tuning (accepted mutations로 LLM 업데이트)
