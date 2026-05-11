# Phase-Conditioned Diffusion Policy

Ant locomotion demonstration에 post-hoc 보행 phase label을 붙이고, Diffusion Policy가 **원하는 phase/frequency trajectory**를 따라 action chunk를 생성할 수 있는지 검증하는 연구용 코드베이스입니다. 프로젝트는 Colab/로컬에서 동일한 `pcdp/` 패키지를 사용하도록 구성되어 있으며, notebook은 실험 흐름을 실행하는 얇은 orchestration layer 역할을 합니다.

## 핵심 아이디어

기본 Diffusion Policy는 최근 observation window만 조건으로 받아 다음 action chunk를 denoising합니다. 이 프로젝트는 여기에 보행 주기 정보를 추가하는 두 가지 확장을 비교합니다.

1. **Vanilla Diffusion Policy**: observation window만 global condition으로 사용합니다.
2. **Periodic Phase Conditioning**: action chunk의 첫 phase `φ₀`를 `(cos φ₀, sin φ₀)`로 인코딩해 global condition에 붙입니다.
3. **Phase Trajectory Conditioning**: action chunk 전체 phase trajectory `φ₀:H`를 step별 `(cos φ, sin φ)`로 인코딩하고, U-Net residual block에 per-step FiLM 방식으로 주입합니다.

현재 main contribution은 **Phase Trajectory Conditioning**입니다.

## 핵심 결과 (TL;DR)

학습 데이터의 평균 frequency `f̄ ≈ 2.023 Hz`, n=20 (Table 1) / n=10 (Table 2), 95% CI.

### Table 1 — In-distribution gait quality

| Model | Reward / step ↑ | Mean x-velocity ↑ | Measured freq (Hz) |
|---|---|---|---|
| Vanilla DP | 0.742 ± 0.087 | 0.167 ± 0.090 | 1.384 ± 0.041 |
| Periodic Phase | 0.870 ± 0.240 | 0.296 ± 0.263 | 1.198 ± 0.090 |
| **Trajectory (ours)** | **1.512 ± 0.136** | **1.138 ± 0.120** | **1.940 ± 0.053** |

→ Trajectory가 vanilla 대비 reward/step **2.04×**, x-velocity **6.81×**, measured frequency도 학습 mean에 가장 가깝게 도달.

### Table 2 — Frequency command tracking (in-dist median)

| f_cmd | Model | \|freq err\| ↓ | PLV ↑ |
|---|---|---|---|
| 2.050 Hz | Periodic | 0.822 ± 0.079 | 0.230 ± 0.051 |
| 2.050 Hz | **Trajectory** | **0.046 ± 0.057** | **0.879 ± 0.081** |

→ Trajectory의 frequency error는 periodic 대비 **18×** 작고, phase locking value (PLV)는 **3.8×** 높음.

전체 결과: [`results/table1_indist_quality.md`](results/table1_indist_quality.md), [`results/table2_frequency_tracking.md`](results/table2_frequency_tracking.md).

## Contributions

이 코드베이스는 다음 세 가지 claim을 입증합니다.

1. **Quality preservation/improvement** — Phase trajectory conditioning은 vanilla baseline 대비 정성적·정량적으로 더 우수한 실보행 정책 (forward velocity ~7×, reward/step 2×)을 학습합니다.
2. **Frequency controllability** — Trajectory conditioning은 in-distribution에서 `|freq err| < 0.06 Hz`, PLV > 0.88의 명령-주파수 추종을 달성하며, periodic (single-step phase) conditioning의 mode collapse를 회피합니다.
3. **Graceful generalization** — 학습 범위보다 느린 frequency (OOD-low)에서 zero-shot으로 동일한 tracking 성능을 보이며, 빠른 영역 (OOD-high)에서는 graceful degradation을 보입니다.

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
│   └── 05_evaluation.ipynb
└── pcdp/
    ├── __init__.py
    ├── configs.py
    ├── data_extraction.py
    ├── data_pipeline.py
    ├── dataset.py
    ├── evaluation.py
    ├── experiment_plots.py
    ├── experiment_runner.py
    ├── models.py
    ├── paths.py
    ├── phase.py
    ├── sampling.py
    └── training.py
```

## 전체 실행 흐름

처음 실행하는 경우 아래 notebook 순서대로 진행하면 됩니다.

```text
01_data_preparation.ipynb
    ↓
02_vanilla_dp.ipynb
    ↓
03_phase_periodic.ipynb
    ↓
04_phase_trajectory.ipynb
    ↓
05_evaluation.ipynb
```

| 순서 | Notebook | 역할 | 주요 산출물 |
|---:|---|---|---|
| 01 | `notebooks/01_data_preparation.ipynb` | Minari Ant dataset에서 phase-label demo를 추출하고, train/val split + train-only normalization stats를 저장합니다. | `data/demos_ant.npz`, `data/norm_stats.npz`, quality/diagnostic figure |
| 02 | `notebooks/02_vanilla_dp.ipynb` | phase condition이 없는 Diffusion Policy baseline을 학습하거나 checkpoint에서 로드합니다. | `checkpoints/vanilla_dp_ckpt.pt`, `figures/vanilla_dp_loss.png` |
| 03 | `notebooks/03_phase_periodic.ipynb` | 첫 phase만 global condition으로 주는 periodic phase baseline을 학습/평가합니다. | `checkpoints/phase_periodic_ckpt.pt`, `figures/phase_periodic_loss.png` |
| 04 | `notebooks/04_phase_trajectory.ipynb` | per-step phase trajectory를 U-Net에 주입하는 main model을 학습/평가합니다. | `checkpoints/phase_trajectory_ckpt.pt`, `figures/phase_trajectory_loss.png` |
| 05 | `notebooks/05_evaluation.ipynb` | 세 모델을 동일 rollout protocol로 비교하고 frequency controllability 결과를 저장합니다. | `results/eval_results.npz`, evaluation figures |

## 모듈별 역할

| 파일 | 설명 |
|---|---|
| `pcdp.paths` | repository root, notebook directory, artifact root와 `data/`, `checkpoints/`, `results/`, `figures/`, `videos/` 디렉터리를 일관되게 해석합니다. `PCDP_ARTIFACT_ROOT` 환경변수 또는 Google Drive mount를 우선 사용합니다. |
| `pcdp.configs` | `vanilla`, `periodic_phase`, `phase_trajectory` named experiment config와 `set_global_seed` helper를 정의합니다. 모델 builder, condition function, scheduler, EMA, checkpoint 이름이 여기에서 연결됩니다. |
| `pcdp.data_extraction` | Minari Ant dataset discovery, episode materialization, Hilbert transform phase extraction, phase quality filtering, demo 저장 및 diagnostic plot 생성을 담당합니다. |
| `pcdp.data_pipeline` | `demos_ant.npz`를 train/val episode로 분할하고 observation/action normalization stats 및 frequency 메타데이터를 `norm_stats.npz`로 저장합니다. |
| `pcdp.dataset` | `AntPhaseDataset`, `encode_phase_cossin` helper, artifact loader, DataLoader builder를 제공합니다. 모든 sample은 `obs`, `action`, `phase`를 반환하므로 vanilla부터 phase-conditioned 모델까지 같은 Dataset을 공유합니다. |
| `pcdp.models` | vanilla/periodic 모델용 Conditional 1D U-Net과 phase trajectory FiLM U-Net을 구현합니다. |
| `pcdp.training` | DDPM epsilon-prediction training loop, EMA validation, checkpoint save/load, training-time condition extraction function을 제공합니다. |
| `pcdp.sampling` | DDIM action chunk sampling, MuJoCo Ant rollout, multi-seed evaluation, sampling-time condition extraction function, NaN-safe `nanmean`/`nanstd` helper를 제공합니다. |
| `pcdp.phase` | target frequency에서 phase trajectory를 생성하고, frequency sweep grid/zone, sweep summary, 그리고 rollout observation에서 measured phase와 PLV/frequency-tracking metric을 추출합니다. |
| `pcdp.evaluation` | checkpoint 로드, in-distribution evaluation, frequency sweep, table markdown export, npz serialization helper를 제공합니다. |
| `pcdp.experiment_runner` | notebook에서 공통으로 쓰는 data/model/scheduler/train-or-load/sample/rollout wrapper를 제공합니다. |
| `pcdp.experiment_plots` | loss curve, sampled chunk, frequency sweep, paper-quality evaluation figure(`plot_paper_figures`)를 생성합니다. |

## 설치

Repository를 clone한 뒤 **editable 모드로 한 번만 설치**하면 `pcdp` 패키지가 import 가능해집니다.

```bash
git clone <repo-url>
cd phase_conditioned_diffusion_policy
pip install -e .
```

`pyproject.toml`에 모든 dependency가 선언되어 있어서 `requirements.txt`를 따로 호출할 필요는 없지만, Colab 등에서 `pip install -r requirements.txt`로도 동일하게 동작합니다.

주요 dependency는 다음과 같습니다.

- `torch`, `numpy`, `matplotlib`
- `diffusers` 및 `huggingface_hub`
- `gymnasium[mujoco]`, `mujoco`, `minari`
- `scipy`, `imageio`, `imageio-ffmpeg`

설치가 끝나면 노트북·스크립트에서 `from pcdp.paths import ...` 같이 일반 Python 패키지처럼 import합니다.

```python
from pcdp.paths import DATA_DIR, CHECKPOINTS_DIR, FIGURES_DIR, ensure_artifact_dirs
ensure_artifact_dirs()
```

Colab에서 MuJoCo 렌더링을 사용하려면 system package를 한 번 설치해야 합니다.

```bash
apt-get update -qq
apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf --quiet
```

또한 노트북 첫 셀에서 OSMesa backend를 지정합니다.

```python
import os
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
```

## Artifact 경로 규칙

`pcdp.paths`는 artifact root를 다음 우선순위로 정합니다.

1. `PCDP_ARTIFACT_ROOT` 환경변수가 설정되어 있으면 해당 경로
2. Google Drive가 `/content/drive/MyDrive`에 mount되어 있으면 `/content/drive/MyDrive/phase_conditioned_diffusion_policy`
3. 그 외에는 `~/phase_conditioned_diffusion_policy_artifacts`

artifact root 아래에는 다음 디렉터리가 사용됩니다.

```text
<artifact_root>/
├── data/
│   ├── demos_ant.npz
│   └── norm_stats.npz
├── checkpoints/
│   ├── vanilla_dp_ckpt.pt
│   ├── phase_periodic_ckpt.pt
│   └── phase_trajectory_ckpt.pt
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
│   ├── phase_periodic_sensitivity.png
│   ├── phase_trajectory_loss.png
│   ├── phase_trajectory_sensitivity.png
│   └── eval_figure1-5_*.png
└── videos/
```

로컬에서 artifact 위치를 명시하고 싶다면 다음처럼 실행합니다.

```bash
export PCDP_ARTIFACT_ROOT=/absolute/path/to/pcdp_artifacts
```

## 데이터 포맷

### `demos_ant.npz`

`01_data_preparation.ipynb`의 Phase 1이 생성하는 raw demonstration artifact입니다.

- `observations`: episode별 observation array
- `actions`: episode별 action array
- `phases`: post-hoc extracted phase `φ ∈ [0, 2π)`
- `episode_lengths`: episode length metadata
- quality/frequency 관련 metadata

### `norm_stats.npz`

`01_data_preparation.ipynb`의 Phase 2가 생성하는 downstream 학습용 metadata입니다.

- `obs_mean`, `obs_std`: observation z-score normalization
- `act_min`, `act_max`, `act_range`: action min-max normalization to `[-1, 1]`
- `obs_horizon`, `pred_horizon`, `action_horizon`
- `obs_dim`, `act_dim`
- `train_eps`, `val_eps`, `seed`
- `freq_window_mean`, `freq_window_std`, `freq_window_min`, `freq_window_max`

Dataset indexing은 episode 내 시점 `t`에 대해 다음 window를 만듭니다.

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

기본 scheduler/training 설정은 `pcdp.configs`에 있습니다.

- DDPM training timesteps: `100`
- DDIM inference steps: `16`
- beta schedule: `squaredcos_cap_v2`
- prediction type: `epsilon`
- default batch size: `256`
- default training epochs: vanilla `100`, phase-conditioned variants `60`
- `cfg.evaluation` rollout default: `max_steps=300`, `n_seeds=5` (quick sanity check)
- **paper-quality evaluation** (`05_evaluation.ipynb`)은 위 default를 override: `max_steps=1000`, `n_seeds=20` (Table 1) / `n_seeds=10` (Table 2)
- **Phase joint** (`ANT_PHASE_JOINT_IDX = 19`): Ant-v5 MuJoCo 모델의 hip joint position 인덱스. 환경 상수이므로 `pcdp.data_extraction`에서 한 곳에서만 정의

## 평가 프로토콜

`05_evaluation.ipynb`는 다음 두 가지 관점으로 모델을 비교합니다.

1. **In-distribution performance (Table 1)**: 학습 데이터의 평균 frequency `f̄`에서 vanilla/periodic/trajectory의 survival, reward/step, x-velocity, measured frequency를 비교합니다 (`n_seeds=20`, `max_steps=1000`).
2. **Frequency controllability sweep (Table 2)**: phase-conditioned 두 모델에 5개 target frequency를 명령하고 measured frequency, |freq error|, PLV, reward/step을 비교합니다 (`n_seeds=10`).

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

`pcdp.experiment_plots.plot_paper_figures`가 5개 paper-quality figure를 생성합니다.

| 파일 | 메시지 |
|---|---|
| `eval_figure1_reward_per_step_comparison.png` | Table 1의 reward/step bar chart (세 모델 비교) |
| `eval_figure2_reward_per_step_vs_freq.png` | 5개 target frequency에서 reward/step 곡선 |
| `eval_figure3_target_vs_measured_freq.png` | command vs measured frequency 산점도 + 대각선 (perfect tracking) |
| `eval_figure4_zone_tracking_metrics.png` | OOD/in-dist zone 별 |freq error| + PLV 막대 |
| `eval_figure5_phase_tracking_timeseries.png` | 대표 rollout의 phase 시계열 + wrapped error (PLV 시각화) |

결과 원본 array는 `results/eval_results.npz`에 저장됩니다.

## 빠른 사용 예시

아래 코드는 artifact가 이미 준비되어 있다는 가정하에 named config로 모델과 DataLoader를 준비하는 최소 예시입니다.

```python
import torch

from pcdp.configs import get_experiment_config, set_global_seed
from pcdp.dataset import load_project_data, build_loaders

set_global_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = get_experiment_config("phase_trajectory")

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

학습/로드/샘플링/rollout은 notebook에서 `experiment_runner.py` wrapper를 통해 실행하는 것을 권장합니다.

## 재현성 메모

- `pcdp.configs`의 `set_global_seed(seed=42)`를 통해 Python/NumPy/PyTorch seed를 고정합니다.
- Data split seed는 `norm_stats.npz`에 함께 저장되어 이후 학습 notebook에서 동일 split을 사용합니다.
- checkpoint에는 model state, EMA state, best EMA state, train/validation loss log, experiment config가 저장됩니다.

## 개발 메모

- Notebook 안에 핵심 로직을 중복 구현하지 말고 `pcdp/` 패키지에 추가한 뒤 notebook에서는 import해서 사용합니다 (notebook은 thin orchestration layer).
- 학습 노트북(02–04)은 학습 + loss curve + (phase-conditioned 변형에 한해) phase sensitivity 시각화까지만 담당합니다. 전체 환경 rollout 평가는 모두 `05_evaluation.ipynb`로 일원화되어 있습니다.
- 새 ablation은 `get_experiment_config(..., training={"num_epochs": 30}, ...)`처럼 section overrides를 넘기면 됩니다. 다만 checkpoint/plot 이름이 같으므로 `artifacts` 섹션도 함께 override해서 파일 충돌을 피해 주세요.
- Phase를 condition으로 쓰는 모델은 raw phase를 직접 넣지 않고 `pcdp.dataset.encode_phase_cossin`이 적용하는 `(cos φ, sin φ)` 인코딩을 사용합니다.
- Action은 학습 중 `[-1, 1]`로 normalization되며 rollout 전 raw MuJoCo action range로 unnormalize됩니다.

### Figure 명명 규칙

| Phase | Prefix | 예시 |
|---|---|---|
| Data preparation (nb01) | `data_` | `data_quality_distribution.png`, `data_phase_advance.png` |
| Training (nb02–04) | `<variant>_` | `vanilla_dp_loss.png`, `phase_trajectory_sensitivity.png` |
| Evaluation (nb05) | `eval_figureN_` | `eval_figure1-5_*.png` |
