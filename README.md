# Phase-Conditioned Diffusion Policy

Ant locomotion 환경에서 phase 정보를 condition으로 사용하는 Diffusion Policy 프로젝트입니다.
현재 메인 contribution은 **Phase Trajectory Conditioning**이며, 이전에 보류했던 phase-consistency sampling branch는 프로젝트 구조에서 제거했습니다.

## Project scope

Implemented variants:

1. **Vanilla Diffusion Policy** — observation window만 global condition으로 사용합니다.
2. **Periodic Phase Conditioning** — chunk 첫 phase를 `(cos φ, sin φ)`로 인코딩해 global condition에 추가합니다.
3. **Phase Trajectory Conditioning** — 전체 action chunk의 phase trajectory를 per-step FiLM condition으로 주입합니다.

Removed from main scope:

- Phase-consistency sampling guidance ablation은 main pipeline에서 제거했습니다.
- 관련 notebook, `K(φ)` 통계 helper, guidance sampler 코드는 레포지토리 구조에서 제외했습니다.

## Notebook order

```text
notebooks/00_data_extraction.ipynb
notebooks/01_data_pipeline.ipynb
notebooks/02_vanilla_dp.ipynb
notebooks/03_phase_periodic.ipynb
notebooks/04_phase_trajectory.ipynb
notebooks/05_evaluation.ipynb
```

## Source layout

```text
src/dataset.py   # Dataset, normalization, Drive data loading, DataLoader helpers
src/models.py    # Vanilla, periodic phase, and phase-trajectory U-Net models
src/training.py  # DDPM training loop, EMA validation, checkpoint helpers, condition extractors
src/sampling.py  # DDIM action sampling, Ant rollout, multi-seed evaluation diagnostics
```

Model checkpoints, generated plots, and evaluation artifacts are expected to live in the Google Drive project directory used by the Colab notebooks.
