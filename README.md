# Phase-Conditioned Diffusion Policy

Ant locomotion demonstration에 post-hoc 보행 phase label을 붙이고, Diffusion Policy가 **원하는 phase/frequency trajectory**를 따라 action chunk를 생성할 수 있는지 검증하는 연구용 코드베이스입니다. 프로젝트는 Colab/로컬에서 동일한 `src/` 모듈을 사용하도록 구성되어 있으며, notebook은 실험 흐름을 실행하는 얇은 orchestration layer 역할을 합니다.

## 핵심 아이디어

기본 Diffusion Policy는 최근 observation window만 조건으로 받아 다음 action chunk를 denoising합니다. 이 프로젝트는 여기에 보행 주기 정보를 추가하는 두 가지 확장을 비교합니다.

1. **Vanilla Diffusion Policy**: observation window만 global condition으로 사용합니다.
2. **Periodic Phase Conditioning**: action chunk의 첫 phase `φ₀`를 `(cos φ₀, sin φ₀)`로 인코딩해 global condition에 붙입니다.
3. **Phase Trajectory Conditioning**: action chunk 전체 phase trajectory `φ₀:H`를 step별 `(cos φ, sin φ)`로 인코딩하고, U-Net residual block에 per-step FiLM 방식으로 주입합니다.

현재 main contribution은 **Phase Trajectory Conditioning**이며, 예전 실험 범위였던 sampling-time phase-consistency guidance는 main pipeline에서 제외되어 있습니다.

## Repository 구성

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
    ├── data_extraction.py
    ├── data_pipeline.py
    ├── dataset.py
    ├── evaluation.py
    ├── experiment_plots.py
    ├── experiment_runner.py
    ├── models.py
    ├── paths.py
    ├── phase.py
    ├── reproducibility.py
    ├── sampling.py
    └── training.py
```

## 전체 실행 흐름

처음 실행하는 경우 아래 notebook 순서대로 진행하면 됩니다.

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

| 순서 | Notebook | 역할 | 주요 산출물 |
|---:|---|---|---|
| 00 | `notebooks/00_data_extraction.ipynb` | Minari Ant dataset 후보를 탐색/다운로드하고, Hilbert transform 기반 phase label을 붙인 demonstration artifact를 생성합니다. | `data/demos_ant_planC.npz`, quality/visualization figure |
| 01 | `notebooks/01_data_pipeline.ipynb` | episode split, normalization statistics, horizon/frequency metadata를 생성하고 data pipeline sanity check를 수행합니다. | `data/norm_stats.npz`, pipeline diagnostic figure |
| 02 | `notebooks/02_vanilla_dp.ipynb` | phase condition이 없는 Diffusion Policy baseline을 학습하거나 checkpoint에서 로드합니다. | `checkpoints/vanilla_dp_ckpt.pt`, `figures/vanilla_dp_loss.png` |
| 03 | `notebooks/03_phase_periodic.ipynb` | 첫 phase만 global condition으로 주는 periodic phase baseline을 학습/평가합니다. | `checkpoints/phase_periodic_ckpt.pt`, `figures/phase_periodic_loss.png` |
| 04 | `notebooks/04_phase_trajectory.ipynb` | per-step phase trajectory를 U-Net에 주입하는 main model을 학습/평가합니다. | `checkpoints/phase_trajectory_ckpt.pt`, `figures/phase_trajectory_loss.png` |
| 05 | `notebooks/05_evaluation.ipynb` | 세 모델을 동일 rollout protocol로 비교하고 frequency/phase controllability 결과를 저장합니다. | `results/eval_results.npz`, evaluation figures |

## 모듈별 역할

| 파일 | 설명 |
|---|---|
| `src/paths.py` | repository root, notebook directory, artifact root와 `data/`, `checkpoints/`, `results/`, `figures/`, `videos/` 디렉터리를 일관되게 해석합니다. `PCDP_ARTIFACT_ROOT` 환경변수 또는 Google Drive mount를 우선 사용합니다. |
| `src/configs.py` | `vanilla`, `periodic_phase`, `phase_trajectory` named experiment config를 정의합니다. 모델 builder, condition function, scheduler, EMA, checkpoint 이름이 여기에서 연결됩니다. |
| `src/data_extraction.py` | Minari Ant dataset discovery, episode materialization, Hilbert transform phase extraction, phase quality filtering, demo 저장 및 diagnostic plot 생성을 담당합니다. |
| `src/data_pipeline.py` | `demos_ant_planC.npz`를 train/val episode로 분할하고 observation/action normalization stats 및 frequency metadata를 `norm_stats.npz`로 저장합니다. |
| `src/dataset.py` | `AntPhaseDataset`, obs/action normalization helper, artifact loader, DataLoader builder를 제공합니다. 모든 sample은 `obs`, `action`, `phase`를 반환하므로 vanilla부터 phase-conditioned 모델까지 같은 Dataset을 공유합니다. |
| `src/models.py` | vanilla/periodic 모델용 Conditional 1D U-Net과 phase trajectory FiLM U-Net을 구현합니다. |
| `src/training.py` | DDPM epsilon-prediction training loop, EMA validation, checkpoint save/load, training-time condition extraction function을 제공합니다. |
| `src/sampling.py` | DDIM action chunk sampling, MuJoCo Ant rollout, multi-seed evaluation, sampling-time condition extraction function을 제공합니다. |
| `src/phase.py` | target frequency에서 phase trajectory를 생성하고, frequency sweep grid/zone 및 sweep summary를 계산합니다. |
| `src/evaluation.py` | checkpoint 로드, in-distribution evaluation, frequency sweep, phase-offset sweep, result serialization helper를 제공합니다. |
| `src/experiment_runner.py` | notebook에서 공통으로 쓰는 data/model/scheduler/train-or-load/sample/rollout wrapper를 제공합니다. |
| `src/experiment_plots.py` | loss curve, sampled chunk, frequency sweep, final evaluation comparison figure를 생성합니다. |
| `src/reproducibility.py` | seed/device 설정과 project path/data summary 출력 helper를 제공합니다. |

## 설치

Python 의존성은 `requirements.txt`에 모여 있습니다.

```bash
pip install -r requirements.txt
```

주요 dependency는 다음과 같습니다.

- `torch`, `numpy`, `matplotlib`
- `diffusers` 및 `huggingface_hub`
- `gymnasium[mujoco]`, `mujoco`, `minari`
- `scipy`, `imageio`, `imageio-ffmpeg`

Colab에서 MuJoCo 렌더링을 사용할 경우 system package 설치가 추가로 필요할 수 있으며, 해당 setup은 `00_data_extraction.ipynb`의 environment setup cell에서 처리합니다.

## Artifact 경로 규칙

`src/paths.py`는 artifact root를 다음 우선순위로 정합니다.

1. `PCDP_ARTIFACT_ROOT` 환경변수가 설정되어 있으면 해당 경로
2. Google Drive가 `/content/drive/MyDrive`에 mount되어 있으면 `/content/drive/MyDrive/phase_conditioned_diffusion_policy`
3. 그 외에는 `~/phase_conditioned_diffusion_policy_artifacts`

artifact root 아래에는 다음 디렉터리가 사용됩니다.

```text
<artifact_root>/
├── data/
│   ├── demos_ant_planC.npz
│   └── norm_stats.npz
├── checkpoints/
│   ├── vanilla_dp_ckpt.pt
│   ├── phase_periodic_ckpt.pt
│   └── phase_trajectory_ckpt.pt
├── results/
├── figures/
└── videos/
```

로컬에서 artifact 위치를 명시하고 싶다면 다음처럼 실행합니다.

```bash
export PCDP_ARTIFACT_ROOT=/absolute/path/to/pcdp_artifacts
```

## 데이터 포맷

### `demos_ant_planC.npz`

`00_data_extraction.ipynb`가 생성하는 raw demonstration artifact입니다.

- `observations`: episode별 observation array
- `actions`: episode별 action array
- `phases`: post-hoc extracted phase `φ ∈ [0, 2π)`
- `episode_lengths`: episode length metadata
- quality/frequency 관련 metadata

### `norm_stats.npz`

`01_data_pipeline.ipynb`가 생성하는 downstream 학습용 metadata입니다.

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

기본 scheduler/training 설정은 `src/configs.py`에 있습니다.

- DDPM training timesteps: `100`
- DDIM inference steps: `16`
- beta schedule: `squaredcos_cap_v2`
- prediction type: `epsilon`
- default batch size: `256`
- default training epochs: vanilla `100`, phase-conditioned variants `60`
- rollout evaluation default: `300` steps, `5` seeds

## 평가 프로토콜

`05_evaluation.ipynb`는 다음 세 가지 관점으로 모델을 비교합니다.

1. **In-distribution performance**: 학습 데이터의 평균 frequency에서 vanilla, periodic, trajectory 모델의 survival/reward를 비교합니다.
2. **Frequency controllability sweep**: phase-conditioned 모델에 여러 target frequency trajectory를 주고, in-distribution/OOD frequency에서 reward와 survival을 비교합니다.
3. **Phase-offset sweep**: 동일 frequency에서 초기 phase offset을 `0`, `π/2`, `π`, `3π/2`로 바꿔 phase condition을 실제로 활용하는지 확인합니다.

결과는 `results/eval_results.npz`로 저장되고, `experiment_plots.py`의 helper로 reward-vs-frequency figure를 만들 수 있습니다.

## 빠른 사용 예시

아래 코드는 artifact가 이미 준비되어 있다는 가정하에 named config로 모델과 DataLoader를 준비하는 최소 예시입니다.

```python
from configs import get_experiment_config
from dataset import load_project_data, build_loaders
from reproducibility import setup_reproducibility

DEVICE = setup_reproducibility(seed=42)
cfg = get_experiment_config("phase_trajectory")

data = load_project_data()
train_loader, val_loader, train_ds, val_ds = build_loaders(
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

- `src/reproducibility.py`의 `setup_reproducibility(seed=42)` 또는 `set_global_seed`를 통해 Python/NumPy/PyTorch seed를 고정합니다.
- Data split seed는 `norm_stats.npz`에 함께 저장되어 이후 학습 notebook에서 동일 split을 사용합니다.
- checkpoint에는 model state, EMA state, best EMA state, train/validation loss log, experiment config가 저장됩니다.

## 개발 메모

- Notebook 안에 핵심 로직을 중복 구현하지 말고 `src/` 모듈에 추가한 뒤 notebook에서는 import해서 사용합니다.
- 새 ablation은 `get_experiment_config(..., overrides)` 또는 `make_ablation_configs`로 만들면 checkpoint/plot 이름 충돌을 줄일 수 있습니다.
- phase를 condition으로 쓰는 모델은 raw phase를 직접 넣지 않고 `(cos φ, sin φ)` 인코딩을 사용합니다.
- action은 학습 중 `[-1, 1]`로 normalization되며 rollout 전 raw MuJoCo action range로 unnormalize됩니다.
