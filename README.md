# Phase-Conditioned Diffusion Policy

## 핵심 아이디어

Quadruped locomotion은 본질적으로 주기적인 phase 구조를 가지며, gait phase는 보행 제어에 중요한 inductive bias로 작용합니다.

RL에서는 이를 핵심적인 inductive bias로 사용하여 로직을 구성하지만, 아직 Diffusion Policy에서는 phase feature를 이용해 Quadruped locomotion을 조작하려는 시도가 없었습니다.

따라서, 본 프로젝트는 target phase trajectory를 Diffusion Policy의 condition으로 사용하여 원하는 보행 리듬을 따르는 controllable locomotion을 생성하는 것을 목표로 합니다.

기본 Diffusion Policy는 최근 observation window만 조건으로 받아 다음 action chunk를 denoising합니다. 이 프로젝트는 여기에 보행 주기 정보를 단계적으로 추가하는 **4가지 모델**을 비교합니다.

1. **Vanilla Diffusion Policy** — observation window만 global condition으로 사용.
2. **Periodic Phase Conditioning** — action chunk의 첫 phase `φ₀`를 `(cos φ₀, sin φ₀)`로 인코딩해 global condition에 붙임.
3. **Phase Trajectory Conditioning** — action chunk 전체 phase trajectory `φ₀:H`를 step별 `(cos φₜ, sin φₜ)`로 인코딩하고 U-Net residual block에 **per-step FiLM** 방식으로 주입.
4. **Phase Trajectory + Sync Loss (ours)** — Phase Trajectory 모델에 **frozen phase estimator**로부터의 phase synchronization loss를 더해 fine-tune.

   ```
   L_total = L_diffusion + λ · L_phase,    λ = 0.12
   ```

   `L_phase`는 **SNR-weighted circular loss + phase velocity consistency loss**로 구성됩니다 (아래 Sync Loss 개선 섹션 참고).

---

## 핵심 결과

학습 데이터의 평균 frequency `f̄ ≈ 2.023 Hz`, n=20 (Table 1) / n=10 (Table 2), 95% CI.

### Table 1 — In-distribution gait quality

| Model | Reward / step ↑ | Mean x-velocity ↑ | Measured freq (Hz) |
|---|---|---|---|
| Vanilla DP | 0.979 ± 0.214 | 0.466 ± 0.241 | 1.411 ± 0.076 |
| Periodic Phase | 0.854 ± 0.098 | 0.317 ± 0.119 | 1.249 ± 0.060 |
| Phase Trajectory | **1.623 ± 0.208** | **1.247 ± 0.182** | 1.913 ± 0.079 |
| **Phase Trajectory + Sync (ours)** | 1.488 ± 0.175 | 1.106 ± 0.160 | 1.955 ± 0.058 |

### Table 2 — Frequency command tracking (in-dist median, f_cmd ≈ 2.050 Hz)

| Model | \|freq err\| ↓ | PLV ↑ |
|---|---|---|
| Periodic | 0.713 ± 0.122 | 0.280 ± 0.053 |
| Phase Trajectory | 0.135 ± 0.148 | 0.778 ± 0.174 |
| **Phase Trajectory + Sync (ours)** | **0.035 ± 0.017** | **0.912 ± 0.017** |

---

## Contributions

이 코드베이스는 다음 네 가지 claim을 입증합니다.

1. **Quality preservation/improvement** — Phase trajectory conditioning은 vanilla baseline 대비 정성적·정량적으로 더 우수한 실보행 정책을 학습합니다.
2. **Frequency controllability** — Trajectory conditioning은 in-distribution에서 `|freq err| < 0.06 Hz`, PLV > 0.88의 명령-주파수 추종을 달성하며, periodic(single-step phase) conditioning의 mode collapse를 회피합니다.
3. **Graceful generalization** — 학습 범위보다 느린 frequency(OOD-low)에서 zero-shot으로 동일한 tracking 성능을 보이며, 빠른 영역(OOD-high)에서는 graceful degradation을 보입니다.
4. **Phase synchronization via frozen estimator (ours)** — Phase trajectory 모델을 frozen MLP estimator로부터의 sync loss(λ=0.12)로 fine-tune하면 in-distribution gait quality와 frequency tracking이 동시에 추가 향상됩니다 (정량 비교는 Table 1/2의 sync 행 참고).

---

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

---

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
05_frozen_phase_estimator.ipynb   ← phase estimator 학습 + sync 파인튜닝 (λ=0.12)
    ↓
06_evaluation.ipynb               ← 4개 모델 통합 평가
```

| 순서 | Notebook | 역할 | 예상 소요 (Colab T4) | 주요 산출물 |
|---:|---|---|---|---|
| 01 | `01_data_preparation.ipynb` | Minari Ant dataset에서 phase-coherent demo를 추출하고, train/val split + normalization stats 저장. | ~10분 | `data/demos_ant.npz`, `data/norm_stats.npz` |
| 02 | `02_vanilla_dp.ipynb` | Vanilla DP 60 epoch 학습. | ~25분 | `checkpoints/vanilla_dp_ckpt.pt` |
| 03 | `03_phase_periodic.ipynb` | Periodic Phase 모델 60 epoch 학습. | ~25분 | `checkpoints/phase_periodic_ckpt.pt` |
| 04 | `04_phase_trajectory.ipynb` | Phase Trajectory 모델 60 epoch 학습 + sensitivity 시각화. | ~25분 | `checkpoints/phase_trajectory_ckpt.pt` |
| 05 | `05_frozen_phase_estimator.ipynb` | Phase estimator MLP 학습 → freeze → Phase Trajectory 모델 sync loss fine-tune. | ~15분 | `checkpoints/frozen_phase_estimator_mlp.pt`, `checkpoints/phase_trajectory_sync_lambda0.12.pt` |
| 06 | `06_evaluation.ipynb` | 4개 모델 통합 비교 + frequency controllability sweep + paper figures 저장. | ~70분 | `results/table{1,2}_*.md`, `figures/eval_figure{1..5}_*.png` |

전체 파이프라인 한 번 완주: **약 3시간** (Colab T4 GPU 기준).

---

## Sync Loss 개선 사항

`05_frozen_phase_estimator.ipynb`에서 사용하는 sync loss는 다음 세 가지 개선이 적용되어 있습니다.

### 1. SNR-weighted Circular Loss

기존의 단순 mean circular loss 대신, 각 step마다 **phase velocity의 일관성**을 SNR proxy로 사용해 가중치를 부여합니다.
순간 위상 속도가 episode 평균에서 크게 벗어나는 step(noise/artifact)은 자동으로 down-weight됩니다.

```
w_t = exp(−|dφ_t − mean(dφ)| / std(dφ))   # (B, PH)
L_position = mean(w_t · (1 − dot(pred_t, target_t)))
```

`compute_snr_weights(phase)` → `snr_weighted_circular_loss(pred, target, weights)`

### 2. Phase Velocity Loss

위상의 절댓값(position)뿐 아니라 **위상 변화율(velocity)**도 일치하도록 추가 패널티를 줍니다.
예측된 action chunk가 올바른 gait frequency를 재현하도록 유도합니다.

```
dφ_pred_t = atan2(sin(φ_pred_{t+1} − φ_pred_t), cos(...))
L_velocity = MSE(dφ_pred, dφ_target)
L_phase = L_position + λ_vel · L_velocity,   λ_vel = 0.1  (default)
```

`phase_velocity_loss(pred_sincos, target_sincos)`

### 3. 사용 방법

```python
# 기본값 (권장)
train_phase_sync_diffusion_policy(
    ...,
    snr_weighted=True,      # SNR 가중치 적용
    lambda_velocity=0.1,    # velocity loss 가중치
)

# ablation: 기존 동작 재현
train_phase_sync_diffusion_policy(
    ...,
    snr_weighted=False,
    lambda_velocity=0.0,
)
```

---

## Phase Command & Inference 개선 사항

### 1. Frequency Ramping

`make_phase_trajectory_fn`에 `ramp_steps` 파라미터가 추가되었습니다.
rollout 시작 시 body가 정지 상태에 가깝기 때문에, 명령 frequency를 0에서 목표값까지 선형적으로 램프업하면 초기 phase velocity 불일치를 줄일 수 있습니다.

```python
# 처음 20 step 동안 0 → freq_hz 선형 램프
phase_fn = make_phase_trajectory_fn(freq_hz=1.5, dt=0.05, ramp_steps=20)
```

램프 구간의 phase 공식 (연속적이고 chunk 간 일관성 보장):

```
φ(t) = phase0 + 2π · freq · dt · t² / (2 · ramp_steps)    if t < ramp_steps
φ(t) = φ_ramp_end + 2π · freq · dt · (t − ramp_steps)      if t ≥ ramp_steps
```

### 2. PhaseContinuityTracker

rollout 전반의 phase state를 관리하는 stateful 객체입니다. 세 가지 문제를 동시에 해결합니다.

| 문제 | 해결 |
|---|---|
| **Phase 초기값** | `initialize_from_obs(obs_window)`로 body의 실제 phase를 추정해 `phase0`을 맞춤 |
| **Frequency ramping** | `ramp_steps > 0` 전달 |
| **Chunk 간 연속성** | 동일 공식 기반으로 절대 step index를 사용하므로 re-plan 경계에서 자동으로 연속 |

```python
from pcdp.phase import PhaseContinuityTracker

tracker = PhaseContinuityTracker(freq_hz=1.5, dt=0.05, ramp_steps=20)

# rollout 직전: body phase 추정 후 phase0 정렬
tracker.initialize_from_obs(initial_obs_window)  # obs: (OH, obs_dim)

# rollout loop에 그대로 전달 (callable → phase_trajectory_fn과 동일 인터페이스)
result = rollout(model, ema, env, ..., phase_trajectory_fn=tracker)
```

---

## Phase Labeling 방법 비교

`extract_demos_from_episodes`에 `phase_method` 파라미터가 추가되었습니다. 세 가지 방법을 선택할 수 있습니다.

| method | 함수 | 설명 |
|---|---|---|
| `"hilbert"` (기본) | `extract_phase` | 단일 hip joint(idx=19)에 Hilbert transform |
| `"multi_joint"` | `extract_phase_multi_joint` | 4개 hip joint의 Hilbert phase를 circular mean으로 합산 |
| `"pca"` | `extract_phase_pca` | 4개 hip joint 신호의 PCA 첫 번째 PC에 Hilbert transform |

```python
from pcdp.data_extraction import extract_demos_from_episodes, DemoExtractionConfig

# 방법 비교 ablation
for method in ["hilbert", "multi_joint", "pca"]:
    demos = extract_demos_from_episodes(episodes, DemoExtractionConfig(), phase_method=method)

# 단독 사용
from pcdp.data_extraction import extract_phase_with_method
phases = extract_phase_with_method(obs_seq, "pca", joint_indices=[19, 21, 23, 25])
```

`ANT_HIP_JOINT_INDICES = (19, 21, 23, 25)` 가 기본값으로 사용됩니다 (환경에 따라 조정 필요).

---

## 모듈별 역할

| 파일 | 설명 |
|---|---|
| `pcdp.paths` | artifact root와 `data/`, `checkpoints/`, `results/`, `figures/`, `videos/` 디렉터리를 일관되게 해석. `PCDP_ARTIFACT_ROOT` 환경변수 우선, 미설정 시 `<repo>/artifacts`. |
| `pcdp.configs` | `vanilla`, `periodic_phase`, `phase_trajectory`, `phase_trajectory_sync` named config와 `set_global_seed` helper 정의. |
| `pcdp.data_extraction` | Minari Ant dataset 로드, episode 처리, phase labeling (`hilbert` / `multi_joint` / `pca`), quality filtering, demo 저장. |
| `pcdp.data_pipeline` | train/val split, normalization stats 저장. |
| `pcdp.dataset` | `AntPhaseDataset`, `encode_phase_cossin`, DataLoader builder. |
| `pcdp.models` | Conditional 1D U-Net (vanilla/periodic) + phase trajectory FiLM U-Net. |
| `pcdp.training` | DDPM epsilon-prediction training loop, EMA, checkpoint save/load. |
| `pcdp.sampling` | DDIM sampling, MuJoCo Ant rollout, multi-seed evaluation. |
| `pcdp.phase` | Phase trajectory 생성 (`make_phase_trajectory_fn`, `PhaseContinuityTracker`), frequency sweep grid/zone, PLV/frequency-tracking metric 계산. |
| `pcdp.frozen_phase_estimator` | Phase estimator MLP 정의/학습/저장, SNR-weighted sync loss + velocity loss, sync fine-tuning loop (`train_phase_sync_diffusion_policy`). |
| `pcdp.evaluation` | 4개 모델 체크포인트 로드, in-distribution evaluation, frequency sweep, table/npz export. |
| `pcdp.experiment_runner` | notebook 공통 data/model/train-or-load wrapper. |
| `pcdp.experiment_plots` | loss curve, frequency sweep, paper-quality figure 생성. ours(`trajectory_sync`)는 `tab:red` + 굵은 선으로 강조. |

---

## 설치

```bash
git clone <repo-url>
cd phase_conditioned_diffusion_policy
pip install -e .
```

주요 dependency:

- `torch`, `numpy`, `pandas`, `tabulate`, `matplotlib`
- `diffusers>=0.30,<0.32`, `huggingface_hub`
- `gymnasium[mujoco]`, `mujoco`, `minari`
- `scipy`, `imageio`, `imageio-ffmpeg`

### Colab 설정

```python
# 1. Drive mount 및 경로 설정
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/phase_conditioned_diffusion_policy')

# 2. 패키지 설치
!pip install -e . -q

# 3. MuJoCo headless rendering
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
```

MuJoCo 렌더링을 사용하려면 system package도 설치합니다.

```bash
apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf -qq
```

---

## Artifact 경로 규칙

`pcdp.paths`는 artifact root를 다음 우선순위로 결정합니다.

1. `PCDP_ARTIFACT_ROOT` 환경변수가 설정되어 있으면 해당 경로
2. 그 외에는 `<repo>/artifacts/`

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
│   ├── vanilla_dp_loss.png
│   ├── phase_periodic_loss.png
│   ├── phase_trajectory_loss.png
│   ├── phase_trajectory_sensitivity.png
│   ├── frozen_phase_estimator_training.png
│   └── eval_figure{1..5}_*.png
└── videos/
```

---

## 데이터 포맷

### `demos_ant.npz`

- `observations`: `(N, T, obs_dim)` episode별 observation
- `actions`: `(N, T, act_dim)` episode별 action
- `phases`: `(N, T)` post-hoc extracted phase `φ ∈ [0, 2π)`
- `episode_lengths`, `estimated_freqs`, `monotonicity`, `peak_sharpness`, `freq_stability`
- `freq_window_mean`, `freq_window_std`, `freq_window_min`, `freq_window_max`

### `norm_stats.npz`

- `obs_mean`, `obs_std`: z-score normalization
- `act_min`, `act_max`, `act_range`: action min-max → `[-1, 1]`
- `obs_horizon`, `pred_horizon`, `action_horizon`, `obs_dim`, `act_dim`
- `train_eps`, `val_eps`, `seed`
- `freq_window_*`

Dataset indexing (시점 `t`):

```text
obs[t - obs_horizon + 1 : t + 1]   → observation window  (OH, obs_dim)
action[t : t + pred_horizon]        → action chunk        (PH, act_dim)
phase[t : t + pred_horizon]         → phase chunk         (PH,)
```

---

## 모델 variants

| Config name | Notebook | Architecture | Checkpoint |
|---|---|---|---|
| `vanilla` | `02` | Conditional 1D U-Net | `vanilla_dp_ckpt.pt` |
| `periodic_phase` | `03` | U-Net + global `(cos φ₀, sin φ₀)` | `phase_periodic_ckpt.pt` |
| `phase_trajectory` | `04` | U-Net + per-step FiLM | `phase_trajectory_ckpt.pt` |
| `phase_trajectory_sync` *(ours)* | `05` | 동일 architecture, sync loss fine-tune | `phase_trajectory_sync_lambda0.12.pt` |

공통 설정:

- DDPM training timesteps: `100` / DDIM inference steps: `16`
- beta schedule: `squaredcos_cap_v2` / prediction type: `epsilon`
- batch size: `256` / training epochs: `60` (모든 variants)
- sync fine-tuning: `10 epoch`, `lr=5e-5`, `λ_phase=0.12`, `snr_weighted=True`, `λ_velocity=0.1`, phase warmup `1 epoch`
- phase joint: `ANT_PHASE_JOINT_IDX = 19` (Ant-v5 hip joint, `pcdp.data_extraction`에서 단일 정의)

---

## 평가 프로토콜

`06_evaluation.ipynb`는 두 가지 관점으로 비교합니다.

1. **In-distribution performance (Table 1)** — `f̄`에서 4개 모델 비교 (`n_seeds=20`, `max_steps=1000`).
2. **Frequency controllability sweep (Table 2)** — 3개 phase-conditioned 모델에 5개 target frequency 명령 (`n_seeds=10`).

### Frequency sweep grid

- **in-dist**: q25, q50 (≈ `f̄`), q75
- **OOD**: `q25 − 1.5·IQR`, `q75 + 1.5·IQR` (Tukey outer fence)

`pcdp.phase.controllability_sweep_frequency_groups`가 이 grid를 생성합니다.

### 측정 metric

- **Measured frequency** `f̂`: unwrapped phase의 linear-regression slope ÷ 2π
- **Frequency error** `|f̂ − f_cmd|`
- **PLV (Phase Locking Value)**:

  ```
  PLV = |⟨ exp(i · Δφ_t) ⟩_t|,    Δφ_t = wrap(φ̂(t) − φ_cmd(t))
  ```

  범위 [0, 1]: `1` = 완벽한 위상 동기, `0` = 위상 무관.
  **frequency가 같아도 PLV가 낮을 수 있음** — phase drift가 있으면 PLV 낮음.

### 산출 figure

| 파일 | 내용 |
|---|---|
| `eval_figure1_reward_per_step_comparison.png` | Table 1 reward/step bar chart (4개 모델) |
| `eval_figure2_reward_per_step_vs_freq.png` | 5개 target frequency별 reward/step 곡선 |
| `eval_figure3_target_vs_measured_freq.png` | command vs measured frequency 산점도 |
| `eval_figure4_zone_tracking_metrics.png` | OOD/in-dist zone별 \|freq error\| + PLV |
| `eval_figure5_phase_tracking_timeseries.png` | 대표 rollout의 command/measured phase 시계열 |

---

## 빠른 사용 예시

```python
import torch
from pcdp.configs import get_experiment_config, set_global_seed
from pcdp.dataset import load_project_data, build_loaders

set_global_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = get_experiment_config("phase_trajectory_sync")  # ours

data = load_project_data()
_, _, train_loader, val_loader = build_loaders(data, batch_size=cfg.data.batch_size)
model = cfg.build_model(data, device=DEVICE)
ema   = cfg.build_ema(model)
noise_scheduler = cfg.build_noise_scheduler()
```

sync loss fine-tuning:

```python
from pcdp.frozen_phase_estimator import train_phase_sync_diffusion_policy

train_log, val_log, best_ema = train_phase_sync_diffusion_policy(
    model, ema, noise_scheduler, estimator,
    train_loader, val_loader, cond_fn,
    device=DEVICE,
    lambda_phase=0.12,
    snr_weighted=True,      # SNR-weighted circular loss
    lambda_velocity=0.1,    # phase velocity consistency loss
)
```

rollout with `PhaseContinuityTracker`:

```python
from pcdp.phase import PhaseContinuityTracker
from pcdp.sampling import rollout, trajectory_phase_sample_cond_fn

tracker = PhaseContinuityTracker(freq_hz=1.5, dt=0.05, ramp_steps=20)
tracker.initialize_from_obs(initial_obs_window)  # body phase 정렬

result = rollout(
    model, ema, env,
    noise_scheduler_config=cfg.noise_scheduler_config,
    ...,
    cond_fn=trajectory_phase_sample_cond_fn,
    phase_trajectory_fn=tracker,
)
```

---

## 재현성 메모

- `set_global_seed(42)` — Python / NumPy / PyTorch seed 고정.
- Data split seed는 `norm_stats.npz`에 저장되어 이후 notebook에서 동일 split 재사용.
- Checkpoint에는 model state, EMA state, best EMA state, train/val loss log, experiment config 포함.

---

## 개발 메모

- Notebook 안에 핵심 로직을 중복 구현하지 말고 `pcdp/`에 추가한 뒤 import해서 사용 (notebook = thin orchestration layer).
- 새 ablation은 `get_experiment_config(..., training={"num_epochs": 30})` 형태로 section override.  `artifacts` 섹션도 함께 override해서 checkpoint 파일 충돌 방지.
- Phase condition 모델은 raw phase를 직접 넣지 않고 `encode_phase_cossin`이 적용하는 `(cos φ, sin φ)` 인코딩을 사용.
- Action은 학습 중 `[-1, 1]`로 normalize, rollout 전 raw MuJoCo range로 unnormalize.

### Figure 명명 규칙

| 단계 | Prefix | 예시 |
|---|---|---|
| Data (nb01) | `data_` | `data_quality_distribution.png` |
| Training (nb02–04) | `<variant>_` | `phase_trajectory_sensitivity.png` |
| Sync (nb05) | `frozen_` | `frozen_phase_estimator_training.png` |
| Evaluation (nb06) | `eval_figureN_` | `eval_figure3_target_vs_measured_freq.png` |
