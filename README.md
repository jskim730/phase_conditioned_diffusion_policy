# Phase-Conditioned Diffusion Policy for Ant Locomotion

Ant locomotion 환경에서 **Diffusion Policy**를 학습하고, 보행 주기 정보를 나타내는 **phase**를 condition으로 주는 프로젝트입니다.  
핵심 목표는 단순히 demonstration action을 모방하는 vanilla policy를 넘어서, 원하는 보행 phase/frequency trajectory를 정책 입력으로 제공했을 때 Ant의 행동이 얼마나 phase에 맞춰 조절되는지 확인하는 것입니다.

현재 프로젝트의 main contribution은 **Phase Trajectory Conditioning**입니다. 이전에 실험했던 phase-consistency sampling guidance는 main pipeline에서 제거했습니다.

---

## 1. 프로젝트 한 줄 요약

> **Ant-v5 / Minari demonstration 데이터에 post-hoc phase label을 붙이고, Vanilla Diffusion Policy → Periodic Phase Conditioning → Phase Trajectory Conditioning 순서로 확장해 phase-conditioned locomotion policy를 평가하는 Colab 중심 연구 코드베이스입니다.**

---

## 2. 현재 구현 범위

| Variant | Notebook | Model / condition 방식 | 목적 |
|---|---|---|---|
| Vanilla Diffusion Policy | `02_vanilla_dp.ipynb` | observation window만 global condition으로 사용 | phase 정보가 없는 baseline |
| Periodic Phase Conditioning | `03_phase_periodic.ipynb` | chunk 첫 phase `φ₀`를 `(cos φ₀, sin φ₀)`로 인코딩해 global condition에 concat | phase scalar를 주기적으로 안정적으로 넣는 1차 확장 |
| Phase Trajectory Conditioning | `04_phase_trajectory.ipynb` | 전체 phase chunk `φ₀:H`를 per-step `(cos φ, sin φ)`로 인코딩하고 U-Net residual block에 FiLM으로 주입 | 프로젝트의 main contribution |
| Evaluation | `05_evaluation.ipynb` | 세 모델을 동일 rollout protocol로 비교 | in-distribution 성능과 frequency controllability 측정 |

### 제거된 범위

- Phase-consistency sampling guidance ablation은 main pipeline에서 제거했습니다.
- 관련 notebook, `K(φ)` 통계 helper, guidance sampler 코드는 현재 구조에서 제외했습니다.
- 따라서 현재 repository는 **학습 시 phase conditioning**과 **rollout/evaluation**에 집중합니다.

---

## 3. 전체 workflow

처음 보는 사람은 아래 순서대로 보면 됩니다.

```text
00_data_extraction.ipynb
    ↓
01_data_pipeline.ipynb
    ↓
02_vanilla_dp.ipynb
    ↓
03_phase_periodic.ipynb
    ↓
04_phase_trajectory.ipynb
    ↓
05_evaluation.ipynb
```

### Notebook별 역할

| 순서 | 파일 | 역할 | 주요 산출물 |
|---:|---|---|---|
| 00 | `notebooks/00_data_extraction.ipynb` | Minari/D4RL 계열 Ant demonstration을 가져오고, post-hoc phase label을 생성 | `demos_ant_planC.npz`, `planC_quality_dist.png`, `planC_demo_visualization.png` |
| 01 | `notebooks/01_data_pipeline.ipynb` | train/val split, normalization stats, horizon metadata 생성 | `norm_stats.npz`, `data_pipeline_sanity.png`, `data_pipeline_phase_advance.png` |
| 02 | `notebooks/02_vanilla_dp.ipynb` | phase 없는 Vanilla Diffusion Policy baseline 학습/로드 | `vanilla_dp_ckpt.pt`, `vanilla_dp_loss.png`, action/chunk diagnostic plots |
| 03 | `notebooks/03_phase_periodic.ipynb` | chunk 첫 phase를 global condition에 넣는 모델 학습/평가 | `phase_periodic_ckpt.pt`, `phase_periodic_loss.png`, frequency sweep plots |
| 04 | `notebooks/04_phase_trajectory.ipynb` | per-step phase trajectory FiLM 모델 학습/평가 | `phase_trajectory_ckpt.pt`, `phase_trajectory_loss.png`, `step3_vs_step4_controllability.png` |
| 05 | `notebooks/05_evaluation.ipynb` | Vanilla / Periodic / Trajectory 세 모델 최종 비교 | `eval_results.npz`, `eval_figure1_reward_vs_freq.png` |

---

## 4. Repository 구조

```text
phase_conditioned_diffusion_policy/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 00_data_extraction.ipynb
│   ├── 01_data_pipeline.ipynb
│   ├── 02_vanilla_dp.ipynb
│   ├── 03_phase_periodic.ipynb
│   ├── 04_phase_trajectory.ipynb
│   └── 05_evaluation.ipynb
└── src/
    ├── configs.py
    ├── dataset.py
    ├── models.py
    ├── training.py
    └── sampling.py
```

### `src/` 모듈 설명

| 파일 | 핵심 내용 |
|---|---|
| `src/configs.py` | named experiment config, scheduler/model/EMA builder, checkpoint path helper, ablation sweep helper |
| `src/dataset.py` | normalization helper, `AntPhaseDataset`, Google Drive project data loader, train/val DataLoader builder |
| `src/models.py` | Diffusion Policy용 Conditional 1D U-Net, Periodic Phase 모델 builder, Phase Trajectory FiLM U-Net |
| `src/training.py` | DDPM epsilon-prediction training loop, EMA validation, checkpoint save/load, training-time condition extractor |
| `src/sampling.py` | DDIM action chunk sampling, MuJoCo Ant rollout, multi-seed evaluation helper, sampling-time condition extractor |

---

## 5. 환경 설정

프로젝트 공통 Python 의존성은 repository root의 `requirements.txt`에서 한 번에 관리합니다. 새 Colab/로컬 런타임에서는 아래 명령으로 동일한 패키지 세트를 설치합니다.

```bash
pip install -r requirements.txt
```

Colab에서 Google Drive의 `PROJECT_DIR` 구조를 사용하는 학습/평가 노트북은 `requirements.txt`도 함께 Drive project directory에 복사되어 있다고 가정합니다. MuJoCo 렌더링에 필요한 system package 설치는 `00_data_extraction.ipynb`의 setup cell에서 별도로 유지합니다.

---

## 6. 데이터와 artifact 위치

노트북은 기본적으로 Google Drive의 아래 디렉터리를 project directory로 사용합니다.

```python
PROJECT_DIR = '/content/drive/MyDrive/quadruped_diffusion_project'
SRC_DIR = f'{PROJECT_DIR}/src'
```

이 repository의 `src/` 파일들은 Colab에서 `PROJECT_DIR/src`로 복사되거나 Drive에 동기화되어 있다고 가정합니다.

### 필수 입력 artifact

| 파일 | 생성 단계 | 설명 |
|---|---|---|
| `demos_ant_planC.npz` | `00_data_extraction.ipynb` | observations, actions, phases, episode_lengths를 포함한 demonstration dataset |
| `norm_stats.npz` | `01_data_pipeline.ipynb` | train-only normalization stats, split indices, horizon metadata, frequency window metadata |

### 모델 checkpoint

| 파일 | 생성 단계 | 설명 |
|---|---|---|
| `vanilla_dp_ckpt.pt` | `02_vanilla_dp.ipynb` | Vanilla DP model + EMA + train/val log |
| `phase_periodic_ckpt.pt` | `03_phase_periodic.ipynb` | Periodic Phase model + EMA + train/val log |
| `phase_trajectory_ckpt.pt` | `04_phase_trajectory.ipynb` | Phase Trajectory model + EMA + train/val log |

### 평가 산출물

| 파일 | 생성 단계 | 설명 |
|---|---|---|
| `eval_results.npz` | `05_evaluation.ipynb` | seed별 survival/reward raw result, sweep result arrays |
| `eval_figure1_reward_vs_freq.png` | `05_evaluation.ipynb` | phase frequency sweep 결과 figure |

---

## 7. 데이터 구조와 horizon 설정

`AntPhaseDataset`은 episode 안의 valid 시점 `t`마다 아래 chunk를 만듭니다.

```text
obs[t - obs_horizon + 1 : t + 1]   → observation window
action[t : t + pred_horizon]       → action chunk
phase[t : t + pred_horizon]        → phase chunk
```

기본 hyperparameter는 `01_data_pipeline.ipynb`에서 저장되며, 이후 notebook들은 `norm_stats.npz`에서 값을 읽어 consistency를 맞춥니다.

| 이름 | 기본값 | 의미 |
|---|---:|---|
| `OBS_HORIZON` | 2 | 현재 step을 포함한 과거 observation window 길이 |
| `PRED_HORIZON` | 16 | diffusion model이 한 번에 생성하는 action chunk 길이 |
| `ACTION_HORIZON` | 8 | rollout에서 sample chunk 중 실제로 실행하는 action 수; 이후 re-plan |
| `NUM_TRAIN_TIMESTEPS` | 100 | DDPM training diffusion step 수 |
| `NUM_INFERENCE_STEPS` | 16 | DDIM inference step 수 |

---

## 8. 모델 설계

### 8.1 Vanilla Diffusion Policy

- 입력 sample: noisy action chunk `(B, pred_horizon, act_dim)`
- global condition: flattened observation window `(B, obs_horizon × obs_dim)`
- 출력: predicted noise `ε̂`
- 목적: phase 정보 없이 observation만으로 action distribution을 학습하는 baseline

### 8.2 Periodic Phase Conditioning

- Vanilla와 같은 `ConditionalUnet1D` architecture를 사용합니다.
- 차이는 global condition에 chunk 첫 phase를 추가하는 것입니다.

```text
global_cond = concat(flatten(obs_window), cos(φ₀), sin(φ₀))
```

raw phase scalar를 그대로 넣지 않고 `(cos, sin)`으로 넣기 때문에 `0`과 `2π` 근처의 discontinuity를 줄입니다.

### 8.3 Phase Trajectory Conditioning — main contribution

- observation window는 global condition으로 유지합니다.
- phase는 전체 chunk trajectory를 per-step condition으로 제공합니다.

```text
per_step_cond[t] = (cos(φ_t), sin(φ_t))
```

`PhaseConditionedUnet1D`는 U-Net의 residual block 안에서:

1. diffusion timestep + observation global condition으로 global FiLM을 적용하고,
2. phase trajectory로 per-step FiLM을 추가 적용합니다.

per-step phase encoder는 zero-init으로 시작해 학습 초기에 기존 Diffusion Policy 동작을 크게 깨지 않도록 설계했습니다.

---

## 9. 학습 방식

공통 training loop는 `src/training.py`의 `train_diffusion_policy()`를 사용합니다.

주요 구성:

- DDPM epsilon-prediction objective
- random diffusion timestep sampling
- MSE loss between predicted noise and sampled noise
- AdamW optimizer
- cosine learning-rate schedule with warmup
- gradient clipping
- EMA validation/checkpointing

모델 variant별 차이는 `cond_fn`만 바뀝니다.

| Variant | training-time condition function |
|---|---|
| Vanilla | `vanilla_cond_fn` |
| Periodic Phase | `periodic_phase_cond_fn` |
| Phase Trajectory | `trajectory_phase_cond_fn` |

### 9.1 Config 중심 실행 패턴

매 notebook에서 `NUM_EPOCHS`, `NUM_TRAIN_TIMESTEPS`, `CKPT_PATH`, `cond_fn` 등을 직접 수정하지 않도록 `src/configs.py`의 named config를 사용합니다. 기본 제공 config는 `vanilla`, `periodic_phase`, `phase_trajectory`입니다.

```python
from configs import get_experiment_config
from dataset import build_loaders, load_project_data
from training import save_checkpoint, train_diffusion_policy

cfg = get_experiment_config(
    'phase_trajectory',
    training={'num_epochs': 60},          # ablation별 override는 여기에서만 변경
    diffusion={'num_inference_steps': 16},
)

data = load_project_data()
train_loader, val_loader = build_loaders(
    data,
    batch_size=cfg.data.batch_size,
    num_workers=cfg.data.num_workers,
)

model = cfg.build_model(data, device=device)
ema = cfg.build_ema(model)
noise_scheduler = cfg.build_noise_scheduler()
cond_fn = cfg.resolve_train_cond_fn()

train_losses, val_log, best_ema_state = train_diffusion_policy(
    model, ema, noise_scheduler, train_loader, val_loader,
    cond_fn=cond_fn, device=device, **cfg.training_kwargs(),
)
save_checkpoint(
    cfg.checkpoint_path(), model, ema, train_losses, val_log,
    best_ema_state=best_ema_state, config=cfg.to_dict(),
)
```

Ablation study는 dotted-key sweep으로 여러 config를 만들 수 있습니다.

```python
from configs import make_ablation_configs

runs = make_ablation_configs('phase_trajectory', {
    'small_unet': {'model.down_dims': (128, 256, 512)},
    'short_train': {'training.num_epochs': 30},
    'more_diffusion_steps': {'diffusion.num_train_timesteps': 200},
})
```

---

## 10. Sampling과 rollout

`src/sampling.py`는 DDIM sampling으로 normalized action chunk를 생성합니다.

1. environment observation을 `obs_mean`, `obs_std`로 normalize
2. 현재 rollout step에서 필요한 phase trajectory 생성
3. DDIM으로 action chunk sample
4. action을 raw scale로 unnormalize
5. 앞 `ACTION_HORIZON` step만 실행
6. 다시 observation window를 업데이트하고 re-plan

즉, evaluation은 receding-horizon control 방식입니다.

---

## 10. Evaluation protocol

`05_evaluation.ipynb`는 세 모델을 같은 protocol로 비교합니다.

### Table 1 — In-distribution performance

- Vanilla
- Periodic Phase
- Phase Trajectory
- 동일 frequency 조건에서 `n=10` deterministic seeds 평가
- metric: survival steps, total reward

### Table 2 / Figure 1 — Frequency controllability sweep

- Periodic Phase vs Phase Trajectory 비교
- 여러 sampling-time phase frequency에서 rollout
- in-distribution frequency window와 OOD frequency에서 성능 변화를 비교
- raw result는 `eval_results.npz`, figure는 `eval_figure1_reward_vs_freq.png`로 저장

---

## 11. 빠른 시작 가이드

### 11.1 Colab에서 실행

각 notebook 상단은 Google Drive mount와 `requirements.txt` 기반 dependency install을 포함합니다.

```python
from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = '/content/drive/MyDrive/quadruped_diffusion_project'
SRC_DIR = f'{PROJECT_DIR}/src'
```

권장 실행 순서:

```text
1. notebooks/00_data_extraction.ipynb
2. notebooks/01_data_pipeline.ipynb
3. notebooks/02_vanilla_dp.ipynb
4. notebooks/03_phase_periodic.ipynb
5. notebooks/04_phase_trajectory.ipynb
6. notebooks/05_evaluation.ipynb
```

이미 Drive에 `demos_ant_planC.npz`, `norm_stats.npz`, checkpoint들이 있다면 각 training notebook의 `TRAIN` flag를 `False`로 두고 checkpoint load/evaluation만 실행할 수 있습니다.

### 11.2 로컬에서 코드 확인

로컬 환경에서는 repository root에서 공통 dependency를 설치한 뒤 코드를 확인할 수 있습니다.

```bash
pip install -r requirements.txt
```

문법 확인은 다음처럼 할 수 있습니다.

```bash
python -m compileall src
```

---

## 12. 현재 프로젝트에서 가장 중요한 파일

처음 코드를 읽는다면 아래 순서로 보는 것을 추천합니다.

1. `src/dataset.py` — 데이터 chunk가 어떻게 만들어지는지 확인
2. `src/models.py` — Vanilla/Periodic/Trajectory 모델 architecture 차이 확인
3. `src/training.py` — condition function과 DDPM training loop 확인
4. `src/sampling.py` — DDIM sampling과 Ant rollout 방식 확인
5. `notebooks/04_phase_trajectory.ipynb` — main contribution 실험 흐름 확인
6. `notebooks/05_evaluation.ipynb` — 최종 비교 protocol 확인

---

## 13. 주의 사항

- 이 repository는 **Colab + Google Drive artifact** 중심으로 구성되어 있습니다.
- checkpoint와 generated plots는 git repository에 포함하지 않고 Drive에 저장하는 구조입니다.
- `README.md`의 artifact 이름은 notebook에서 사용하는 기본 path/name 기준입니다.
- phase-consistency sampling 관련 산출물이나 코드가 필요하다면 현재 main pipeline에는 없으므로, 별도 archive/이전 commit에서 확인해야 합니다.

---

## 14. 요약

이 프로젝트는 다음 질문에 답하기 위한 구현입니다.

> “Ant locomotion Diffusion Policy에 phase를 condition으로 주면, 보행 주기/frequency를 더 잘 제어할 수 있는가?”

현재 답을 찾기 위한 핵심 비교는 다음 세 모델입니다.

```text
Vanilla DP
vs Periodic Phase Conditioning
vs Phase Trajectory Conditioning
```

그리고 최종적으로 `05_evaluation.ipynb`에서 reward, survival, frequency controllability를 같은 protocol로 측정합니다.
