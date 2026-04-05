# R_Q Self-Evolving Problem Generation

하나의 모델이 Questioner(문제 프로그램 진화)와 Solver(문제 풀이) 역할을 동시에 수행하며 Self-Evolving하는 프레임워크. R_Q = p(1-p) · H̄ 를 fitness로 사용하여 역방향 구성 기반 문제 생성 프로그램을 MAP-Elites로 진화시키고, 생성된 자연어 수학 문제로 같은 모델을 veRL GRPO 학습합니다.

## 핵심 아이디어

- **Self-Evolving**: 하나의 모델이 문제를 만들고(mutator) 풀면서(solver) 동시에 강해짐
- **프로그램 진화**: 개별 문제가 아닌 문제 생성 프로그램(Python 함수)을 진화 단위로 사용
- **역방향 구성**: 정답을 먼저 선택 → 같은 변수로 자연어 word problem 구성 → 정답 보장
- **R_Q fitness**: Learnability p(1-p)와 gradient richness H̄를 동시에 포착
- **MAP-Elites**: H 구간 × embedding 클러스터 grid로 다양성 구조적 보장

## 프로젝트 구조

```
rq_evolve/
├── run.py                          # CLI 진입점 (--model 단일 모델)
├── requirements.txt
├── rq_questioner/                  # 핵심 모듈
│   ├── pipeline.py                 # Algorithm 4.1 Self-Evolving loop
│   ├── program.py                  # ProblemProgram (진화 단위)
│   ├── mutator.py                  # mutation 프롬프트 정의
│   ├── map_elites.py               # H × embedding cluster grid
│   ├── verifier.py                 # SymPy 대입 검증
│   ├── entropy.py                  # H 측정 유틸리티
│   └── rq_score.py                 # R_Q = p(1-p) · H
├── evaluation/                     # 벤치마크 평가 (R-Zero 방식)
│   └── __init__.py                 # GSM8K, MATH-500, AIME-2024
├── seed_programs/                  # 자연어 word problem 시드 12개
│   ├── 01_age.py                   # 나이 문제
│   ├── 02_rectangle.py             # 직사각형 넓이
│   ├── 03_speed.py                 # 속도/거리/시간
│   ├── 04_mixture.py               # 혼합 농도
│   ├── 05_gcd.py                   # 최대공약수
│   ├── 06_remainder.py             # 나머지
│   ├── 07_pythagorean.py           # 피타고라스
│   ├── 08_circle.py                # 원의 넓이
│   ├── 09_handshake.py             # 조합 (악수)
│   ├── 10_probability.py           # 확률
│   ├── 11_sequence.py              # 등차급수
│   └── 12_work_rate.py             # 일률 문제
├── configs/
│   └── grpo_config.yaml            # veRL GRPO 설정
└── scripts/
    ├── run_full.sh                 # 전체 파이프라인
    └── test_local.py               # 5/5 PASS (GPU 불필요)
```

## 실행 방법

```bash
# 1. 로컬 검증 (GPU 불필요)
python scripts/test_local.py

# 2. 전체 파이프라인 (GPU 필요)
python run.py full \
    --model Qwen/Qwen2.5-7B-Instruct \
    --seed_dir ./seed_programs \
    --output_dir ./rq_output \
    --num_epochs 5 \
    --num_generations 100
```

## 알고리즘 (제안서 4.1절)

```
Epoch loop (Self-Evolving):
  Phase 1 (vLLM 로드):
    - 벤치마크 평가 (GSM8K, MATH-500, AIME-2024)
    - N세대 진화:
      1. MAP-Elites에서 parent 샘플링
      2. 같은 모델(vLLM)이 프로그램 mutation 생성
      3. 프로그램 실행 → 자연어 (문제, 정답) 생성
      4. SymPy 검증 → 실패 시 폐기
      5. H pre-filter: vLLM(n=1, logprobs=1) → 엔트로피 측정
      6. G rollouts: vLLM(n=16) → p̂ 추정
      7. R_Q = p(1-p) · H̄ → MAP-Elites 갱신
    - Champion → 새 seed → parquet 저장
    - vLLM 해제 (GPU 해방)

  Phase 2 (veRL subprocess):
    - veRL GRPO로 같은 모델 학습
    - Checkpoint 저장

  다음 Epoch:
    - 새 checkpoint로 vLLM 로드
    - 더 잘 진화 + 더 잘 풀기 → 선순환
```

## 평가 벤치마크

R-Zero 논문과 동일:
- **GSM8K**: pass@1 (기본 200문제, -1 for full 1319)
- **MATH-500**: pass@1 (기본 100문제, -1 for full 500)
- **AIME-2024**: mean@32 (30문제 전체)

각 epoch 시작 시 Phase 1에서 자동 평가. Epoch 0 = baseline.

## 구현 상태

| 구성 요소 | 상태 |
|---|---|
| 단일 모델 Self-Evolving 구조 | ✅ 구현 |
| 자연어 word problem 시드 (12개) | ✅ 구현 |
| vLLM 기반 H 측정 (logprobs) | ✅ 구현 |
| vLLM 기반 G rollouts (n=16) | ✅ 구현 |
| MAP-Elites (H × embedding) | ✅ 구현 |
| SymPy 역방향 구성 검증 | ✅ 구현 |
| Phase 분리 (vLLM ↔ veRL) | ✅ 구현 |
| 벤치마크 평가 (GSM8K/MATH/AIME) | ✅ 구현 |
| veRL GRPO 연동 (subprocess) | ✅ 구현 |
| 커스텀 reward function | ⚠️ veRL 기본 rule-based 사용 |
| Multi-GPU 분산 진화 | ❌ 미구현 |
| Mutator fine-tuning | ❌ 미구현 |
