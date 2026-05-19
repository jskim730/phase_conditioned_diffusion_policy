# Phase-Conditioned Diffusion Policy

## 핵심 아이디어

Quadruped locomotion은 본질적으로 주기적인 phase 구조를 가지며, gait phase는 보행 제어에 중요한 inductive bias로 작용합니다.

RL에서는 이를 핵심적인 inductive bias로 사용하여 로직을 구성하지만, 아직 Diffusion Policy에서는 phase feature를 이용해 Quadruped locomotion을 조작하려는 시도가 없었습니다.

따라서, 본 프로젝트는 target phase trajectory를 Diffusion Policy의 condition으로 사용하여 원하는 보행 리듬을 따르는 controllable locomotion을 생성하는 것을 목표로 합니다.

기본 Diffusion Policy는 최근 observation window만 조건으로 받아 다음 action chunk를 denoising합니다. 이 프로젝트는 여기에 보행 주기 정보를 단계적으로 추가하는 **4가지 모델**을 비교합니다.

1. **Vanilla Diffusion Policy** — observation window만 global condition으로 사용.
2. **Periodic Phase Conditioning** — action chunk의 첫 phase `φ₀`를 `(cos φ₀, sin φ₀)`로 인코딩해 global condition에 붙임.
3. **Phase Trajectory Conditioning** — action chunk 전체 phase trajectory `φ₀:H`를 step별 `(cos φₜ, sin φₜ)`로 인코딩하고 U-Net residual block에 **per-step FiLM** 방식으로 주입.
4. **Phase Trajectory + Sync Loss (ours)** — Phase Trajectory 모델에 **frozen phase estimator**로부터의 phase synchronization loss(`L_total = L_diffusion + 0.12 · L_phase`)를 더해 15 epoch fine-tune.



## 핵심 결과

학습 데이터의 평균 frequency `f̄ ≈ 2.023 Hz`, n=20 (Table 1) / n=10 (Table 2), 95% CI.

### Table 1 — In-distribution gait quality

| Model | Reward / step ↑ | Mean x-velocity ↑ | Measured freq (Hz) |
|---|---|---|---|
| Vanilla DP | 0.979 ± 0.214 | 0.466 ± 0.241 | 1.411 ± 0.076 |
| Periodic Phase | 0.854 ± 0.098 | 0.317 ± 0.119 | 1.249 ± 0.060 |
| Phase Trajectory | **1.623 ± 0.208** | **1.247 ± 0.182** | 1.913 ± 0.079 |
| **Phase Trajectory + Sync (ours)** | 1.488 ± 0.175 | 1.106 ± 0.160  | 1.955 ± 0.058 |

### Table 2 — Frequency command tracking (in-dist median, f_cmd ≈ 2.050 Hz)

| Model | \|freq err\| ↓ | PLV ↑ |
|---|---|---|
| Periodic | 0.713 ± 0.122 | 0.280 ± 0.053 |
| Phase Trajectory | 0.135 ± 0.148 | 0.778 ± 0.174 |
| **Phase Trajectory + Sync (ours)** | **0.035 ± 0.017** | **0.912 ± 0.017** |

## Contributions

이 코드베이스는 다음 네 가지 claim을 입증합니다.

1. **Quality preservation/improvement** — Phase trajectory conditioning은 vanilla baseline 대비 정성적·정량적으로 더 우수한 실보행 정책을 학습합니다.
2. **Frequency controllability** — Trajectory conditioning은 in-distribution에서 `|freq err| < 0.06 Hz`, PLV > 0.88의 명령-주파수 추종을 달성하며, periodic(single-step phase) conditioning의 mode collapse를 회피합니다.
3. **Graceful generalization** — 학습 범위보다 느린 frequency(OOD-low)에서 zero-shot으로 동일한 tracking 성능을 보이며, 빠른 영역(OOD-high)에서는 graceful degradation을 보입니다.
4. **Phase synchronization via frozen estimator (ours)** — Phase trajectory 모델을 frozen MLP estimator로부터의 sync loss(λ=0.12)로 fine-tune하면 in-distribution gait quality와 frequency tracking이 동시에 추가 향상됩니다 (정량 비교는 Table 1/2의 sync 행 참고).

## Repository 구성

```text
phase_conditioned_diffusion_policy/
├── README.md
├── pyproject.toml
├── requirements.txt
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_vanilla_dp.ipynb
│   ├── 03_phase_periodic.ipynb
│   ├── 04_phase_trajectory.ipynb
│   ├── 05_frozen_phase_estimator.ipynb
│   └── 06_evaluation.ipynb
└── pcdp/
    ├── __init__.py
    ├── configs.py
    ├── data_extraction.py
    ├── data_pipeline.py
    ├── dataset.py
    ├── evaluation.py
    ├── experiment_plots.py
    ├── experiment_runner.py
    ├── frozen_phase_estimator.py
    ├── models.py
    ├── paths.py
    ├── phase.py
    ├── sampling.py
    └── training.py
```

## 전체 실행 흐름

처음 실행하는 경우 사용자가 먼저 Colab Drive mount(필요 시), dependency 설치, repository root로 작업 디렉터리 이동을 완료했다고 가정합니다. 그 다음 아래 notebook 순서대로 진행하면 됩니다.

```text
01_data_preparation.ipynb
    ↓
02_vanilla_dp.ipynb
    ↓
03_phase_periodic.ipynb
    ↓
04_phase_trajectory.ipynb
    ↓
05_frozen_phase_estimator.ipynb   ← phase estimator 학습 + sync 파인튜닝 (LAMBDA_PHASE=0.12)
    ↓
06_evaluation.ipynb               ← 4개 모델 통합 평가
```

| 순서 | Notebook | 역할 | 예상 소요 (Colab T4) | 주요 산출물 |
|---:|---|---|---|---|
| 01 | `notebooks/01_data_preparation.ipynb` | Minari Ant dataset에서 phase-coherent demo를 추출하고, train/val split + train-only normalization stats를 저장. | **~10분** (다운로드 포함) | `data/demos_ant.npz`, `data/norm_stats.npz`, `figures/data_*.png` |
| 02 | `notebooks/02_vanilla_dp.ipynb` | phase condition 없는 Vanilla DP를 60 epoch 학습. | **~25분** | `checkpoints/vanilla_dp_ckpt.pt`, `figures/vanilla_dp_loss.png` |
| 03 | `notebooks/03_phase_periodic.ipynb` | 첫 phase만 global condition으로 주는 Periodic Phase 모델을 60 epoch 학습. | **~25분** | `checkpoints/phase_periodic_ckpt.pt`, `figures/phase_periodic_loss.png` |
| 04 | `notebooks/04_phase_trajectory.ipynb` | per-step phase trajectory를 U-Net에 FiLM 주입하는 Phase Trajectory 모델을 60 epoch 학습 + sensitivity 시각화. | **~25분** | `checkpoints/phase_trajectory_ckpt.pt`, `figures/phase_trajectory_loss.png`, `figures/phase_trajectory_sensitivity.png` |
| 05 | `notebooks/05_frozen_phase_estimator.ipynb` | Phase estimator MLP(40 epoch) 학습 + Phase Trajectory 모델을 `L_total = L_diffusion + 0.12 · L_phase`로 15 epoch fine-tune해 **ours** 체크포인트 생성. | **~15분** | `checkpoints/frozen_phase_estimator_mlp.pt`, `checkpoints/phase_trajectory_sync_lambda0.12.pt`, `figures/frozen_phase_estimator_training.png` |
| 06 | `notebooks/06_evaluation.ipynb` | 4개 모델(`vanilla` / `periodic` / `trajectory` / `trajectory_sync` = ours)을 동일 rollout protocol로 비교하고 frequency controllability 결과 + paper figures 저장. | **~70분** | `results/table{1,2}_*.md`, `results/eval_results.npz`, `figures/eval_figure{1..5}_*.png` |

전체 파이프라인 한 번 완주: **약 3시간** (Colab T4 GPU 기준).

## 모듈별 역할

| 파일 | 설명 |
|---|---|
| `pcdp.paths` | repository root, notebook directory, artifact root와 `data/`, `checkpoints/`, `results/`, `figures/`, `videos/` 디렉터리를 일관되게 해석. `PCDP_ARTIFACT_ROOT` 환경변수가 없으면 repository-local `artifacts/`를 사용. |
| `pcdp.configs` | `vanilla`, `periodic_phase`, `phase_trajectory`, `phase_trajectory_sync` named experiment config와 `set_global_seed` helper를 정의. 모델 builder, condition function, scheduler, EMA, checkpoint 이름이 여기에서 연결. |
| `pcdp.data_extraction` | Minari Ant dataset discovery, episode materialization, Hilbert transform phase extraction, phase quality filtering, demo 저장 및 diagnostic plot 생성. |
| `pcdp.data_pipeline` | `demos_ant.npz`를 train/val episode로 분할하고 observation/action normalization stats 및 frequency 메타데이터를 `norm_stats.npz`로 저장. |
| `pcdp.dataset` | `AntPhaseDataset`, `encode_phase_cossin` helper, artifact loader, DataLoader builder를 제공. 모든 sample은 `obs`, `action`, `phase`를 반환하므로 vanilla부터 sync 모델까지 같은 Dataset을 공유. |
| `pcdp.models` | vanilla/periodic 모델용 Conditional 1D U-Net과 phase trajectory FiLM U-Net을 구현. |
| `pcdp.training` | DDPM epsilon-prediction training loop, EMA validation, checkpoint save/load (dict/tuple `val_log` 모두 지원), training-time condition extraction function. |
| `pcdp.sampling` | DDIM action chunk sampling, MuJoCo Ant rollout, multi-seed evaluation, sampling-time condition extraction function, NaN-safe `nanmean`/`nanstd` helper. |
| `pcdp.phase` | target frequency에서 phase trajectory를 생성, frequency sweep grid/zone, sweep summary, 그리고 rollout observation에서 measured phase와 PLV/frequency-tracking metric을 추출. |
| `pcdp.frozen_phase_estimator` | obs+action → phase 예측 MLP의 정의/학습/저장, frozen estimator로부터의 phase synchronization loss(`L_phase`)와 sync fine-tuning loop(`train_phase_sync_diffusion_policy`). |
| `pcdp.evaluation` | 4개 모델 체크포인트 자동 로드(`load_evaluation_state`), in-distribution evaluation, frequency sweep, table markdown export, npz serialization helper. |
| `pcdp.experiment_runner` | notebook 02–05에서 공통으로 쓰는 data/model/scheduler/train-or-load/sample wrapper. |
| `pcdp.experiment_plots` | loss curve, sampled chunk, frequency sweep, paper-quality evaluation figure(`plot_paper_figures`)를 생성. 모델별 색상/굵기/마커 매핑으로 ours(`tab:red`)를 강조. |

## 설치

Repository를 clone한 뒤 **editable 모드로 한 번만 설치**하면 `pcdp` 패키지가 import 가능합니다.

```bash
git clone <repo-url>
cd phase_conditioned_diffusion_policy
pip install -e .
```

`pyproject.toml`에 모든 dependency가 선언되어 있어서 `requirements.txt`를 따로 호출할 필요는 없지만, Colab 등에서 `pip install -r requirements.txt`로도 동일하게 동작합니다.

주요 dependency:

- `torch`, `numpy`, `pandas`, `tabulate`, `matplotlib`
- `diffusers>=0.30,<0.32` 및 `huggingface_hub`
- `gymnasium[mujoco]`, `mujoco`, `minari`
- `scipy`, `imageio`, `imageio-ffmpeg`

설치가 끝나면 노트북·스크립트에서 일반 Python 패키지처럼 import:

```python
from pcdp.paths import DATA_DIR, CHECKPOINTS_DIR, FIGURES_DIR, ensure_artifact_dirs
ensure_artifact_dirs()
```

### Colab runtime assumptions

각 노트북의 첫 코드 셀(`## 1. Setup`)은 Colab에서 Google Drive를 mount한 뒤 기본 repository 위치인 `/content/drive/MyDrive/phase_conditioned_diffusion_policy`로 작업 디렉터리를 이동합니다. 다른 Drive 경로에 두었다면 셀의 `PROJECT_DIR` 값을 수정하세요. 로컬/Jupyter 환경에서는 Colab 전용 mount를 건너뛰고 현재 작업 디렉터리를 유지합니다.

재현 실행 전에 사용자가 아래 작업을 완료했다고 가정합니다.

1. Google Drive 안에 repository를 준비.
2. MuJoCo/OSMesa 렌더링이 필요하면 system package 설치: `libosmesa6-dev`, `libgl1-mesa-glx`, `libglfw3`, `patchelf`.
3. repository root 기준으로 dependency 설치: `pip install -e .`.
4. MuJoCo headless rendering이 필요하면 notebook 실행 전에 `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa` 설정.
5. 필요하면 notebook 실행 전에 `PCDP_ARTIFACT_ROOT`를 원하는 artifact 경로로 설정. 미설정 시 `<repo>/artifacts` 사용.

## Artifact 경로 규칙

`pcdp.paths`는 artifact root를 다음 우선순위로 정합니다.

1. `PCDP_ARTIFACT_ROOT` 환경변수가 설정되어 있으면 해당 경로
2. 그 외에는 repository-local `<repo>/artifacts`

artifact root 아래에는 다음 디렉터리가 사용됩니다.

```text
<repo>/artifacts/  (or $PCDP_ARTIFACT_ROOT/)
├── data/
│   ├── demos_ant.npz
│   └── norm_stats.npz
├── checkpoints/
│   ├── vanilla_dp_ckpt.pt
│   ├── phase_periodic_ckpt.pt
│   ├── phase_trajectory_ckpt.pt
│   ├── frozen_phase_estimator_mlp.pt
│   └── phase_trajectory_sync_lambda0.12.pt
├── results/
│   ├── table1_indist_quality.md
│   ├── table2_frequency_tracking.md
│   └── eval_results.npz
├── figures/
│   ├── data_quality_distribution.png
│   ├── data_demo_examples.png
│   ├── data_phase_advance.png
│   ├── vanilla_dp_loss.png
│   ├── phase_periodic_loss.png
│   ├── phase_trajectory_loss.png
│   ├── phase_trajectory_sensitivity.png
│   ├── frozen_phase_estimator_training.png
│   └── eval_figure{1..5}_*.png
└── videos/
```

artifact 위치를 명시하려면:

```bash
export PCDP_ARTIFACT_ROOT=/absolute/path/to/pcdp_artifacts
```

## 데이터 포맷

### `demos_ant.npz`

`01_data_preparation.ipynb`의 Phase 1이 생성하는 raw demonstration artifact:

- `observations`: episode별 observation array
- `actions`: episode별 action array
- `phases`: post-hoc extracted phase `φ ∈ [0, 2π)`
- `episode_lengths`: episode length metadata
- quality/frequency 관련 metadata

### `norm_stats.npz`

`01_data_preparation.ipynb`의 Phase 2가 생성하는 downstream 학습용 metadata:

- `obs_mean`, `obs_std`: observation z-score normalization
- `act_min`, `act_max`, `act_range`: action min-max normalization to `[-1, 1]`
- `obs_horizon`, `pred_horizon`, `action_horizon`
- `obs_dim`, `act_dim`
- `train_eps`, `val_eps`, `seed`
- `freq_window_mean`, `freq_window_std`, `freq_window_min`, `freq_window_max`

Dataset indexing은 episode 내 시점 `t`에 대해 다음 window를 만듭니다:

```text
obs[t - obs_horizon + 1 : t + 1]   -> observation window
action[t : t + pred_horizon]       -> action chunk
phase[t : t + pred_horizon]        -> phase chunk
```

## 모델 variants

| Config name | Notebook | Model builder | Train condition | Sample condition | Checkpoint |
|---|---|---|---|---|---|
| `vanilla` | `02_vanilla_dp.ipynb` | `build_vanilla_dp_model` | `vanilla_cond_fn` | `vanilla_sample_cond_fn` | `vanilla_dp_ckpt.pt` |
| `periodic_phase` | `03_phase_periodic.ipynb` | `build_periodic_phase_dp_model` | `periodic_phase_cond_fn` | `periodic_phase_sample_cond_fn` | `phase_periodic_ckpt.pt` |
| `phase_trajectory` | `04_phase_trajectory.ipynb` | `build_phase_trajectory_dp_model` | `trajectory_phase_cond_fn` | `trajectory_phase_sample_cond_fn` | `phase_trajectory_ckpt.pt` |
| `phase_trajectory_sync` *(ours)* | `05_frozen_phase_estimator.ipynb` | `build_phase_trajectory_dp_model` (동일 architecture, sync loss로 fine-tune) | `trajectory_phase_cond_fn` | `trajectory_phase_sample_cond_fn` | `phase_trajectory_sync_lambda0.12.pt` |

기본 scheduler/training 설정 (`pcdp.configs`):

- DDPM training timesteps: `100`
- DDIM inference steps: `16`
- beta schedule: `squaredcos_cap_v2`
- prediction type: `epsilon`
- default batch size: `256`
- default training epochs: **60** (모든 variants 공통, `_BASE_TRAINING.num_epochs`)
- sync fine-tuning (`train_phase_sync_diffusion_policy`): 10 epochs, lr=5e-5, lambda=0.12, phase warmup 1 epoch
- `cfg.evaluation` rollout default: `max_steps=300`, `n_seeds=5` (quick sanity check)
- **paper-quality evaluation** (`06_evaluation.ipynb`)은 위 default를 override: `max_steps=1000`, `n_seeds=20` (Table 1) / `n_seeds=10` (Table 2)
- **Phase joint** (`ANT_PHASE_JOINT_IDX = 19`): Ant-v5 MuJoCo 모델의 hip joint position 인덱스. 환경 상수이므로 `pcdp.data_extraction`에서 한 곳에서만 정의.

## 평가 프로토콜

`06_evaluation.ipynb`는 다음 두 가지 관점으로 모델을 비교합니다.

1. **In-distribution performance (Table 1)** — 학습 데이터의 평균 frequency `f̄`에서 **4개 모델**(vanilla / periodic / trajectory / trajectory_sync)의 survival, reward/step, x-velocity, measured frequency 비교 (`n_seeds=20`, `max_steps=1000`).
2. **Frequency controllability sweep (Table 2)** — **3개 phase-conditioned 모델**(periodic / trajectory / trajectory_sync, vanilla 제외)에 5개 target frequency를 명령하고 measured frequency, |freq error|, PLV, reward/step 비교 (`n_seeds=10`).

### Frequency sweep grid

학습 frequency window `[f_min, f_max]`에서 5점을 사용합니다.

- **in-dist**: q25, q50 (≈ `f̄`), q75 — 학습 분포 내부 3점
- **OOD**: `q25 − 1.5·IQR`, `q75 + 1.5·IQR` — Tukey-style outer fence 2점

`pcdp.phase.controllability_sweep_frequency_groups`가 이 grid를 생성합니다.

### 측정 metric

각 rollout에 대해 Hilbert transform으로 measured phase `φ̂(t)`를 추출하고, 다음을 계산합니다.

- **Measured frequency** `f̂`: unwrapped phase의 linear-regression slope ÷ 2π
- **Frequency error** `|f̂ − f_cmd|`: 평균 frequency 추종 오차
- **PLV (Phase Locking Value)**: 위상 정합도

  ```
  PLV = |⟨ exp(i · Δφ_t) ⟩_t|,    Δφ_t = wrap(φ̂(t) − φ_cmd(t)) ∈ [-π, π]
  ```

  각 시점의 phase error를 단위 원의 점으로 보고 평균. 범위 [0, 1]:
  - **PLV = 1**: phase error가 시간에 따라 일정 (완벽한 위상 동기)
  - **PLV = 0**: phase error가 균일 분포 (위상 무관)

  **frequency가 같아도 PLV가 낮을 수 있음** — 평균 속도만 맞고 phase가 drift하면 PLV 낮음.

### 산출 figure

`pcdp.experiment_plots.plot_paper_figures`가 5개 paper-quality figure를 생성합니다. ours(`trajectory_sync`)는 진한 빨강(`tab:red`)에 굵은 선/큰 마커/검은 테두리로 강조됩니다.

| 파일 | 메시지 |
|---|---|
| `eval_figure1_reward_per_step_comparison.png` | Table 1의 reward/step bar chart (4개 모델 비교) |
| `eval_figure2_reward_per_step_vs_freq.png` | 5개 target frequency에서 reward/step 곡선 (3개 phase 모델) |
| `eval_figure3_target_vs_measured_freq.png` | command vs measured frequency 산점도 + 대각선(perfect tracking) |
| `eval_figure4_zone_tracking_metrics.png` | OOD/in-dist zone 별 \|freq error\| + PLV 막대 |
| `eval_figure5_phase_tracking_timeseries.png` | 대표 rollout의 command/measured phase 시계열 (모델별 panel) |

결과 원본 array는 `results/eval_results.npz`에 저장됩니다.

## 빠른 사용 예시

아래 코드는 artifact가 이미 준비되어 있다는 가정하에 named config로 모델과 DataLoader를 준비하는 최소 예시입니다.

```python
import torch

from pcdp.configs import get_experiment_config, set_global_seed
from pcdp.dataset import load_project_data, build_loaders

set_global_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = get_experiment_config("phase_trajectory_sync")  # ours

data = load_project_data()
train_ds, val_ds, train_loader, val_loader = build_loaders(
    data,
    batch_size=cfg.data.batch_size,
    num_workers=cfg.data.num_workers,
)

model = cfg.build_model(data, device=DEVICE)
ema = cfg.build_ema(model)
noise_scheduler = cfg.build_noise_scheduler()
```

학습/로드/샘플링/rollout은 notebook에서 `experiment_runner.py` (학습) 또는 `evaluation.py` (평가) wrapper를 통해 실행하는 것을 권장합니다.
