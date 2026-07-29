# GEAR Implementation

This directory contains the implementation and evaluation code for:

> **GEAR: Geometry-Guided Evidence for Action Recovery in  
> Inference-Time Defense of Vision-Language-Action Models**

Accepted by **CICAI 2026**.

For the project overview, main results, and citation information, see the
[root README](../README.md).

---

## Code Version

The final paper experiments correspond to:

```text
Repository: woruqingshan/RoboticAttack_Defense
Branch: release/cicai2026
Experimental commit: d1dc5a88e27d8293b7add03561c12b8519bc7774
```

The repository `main` branch follows an earlier development path and does not
exactly match the final GEAR mechanism reported in the paper.

For paper reproduction, use:

```bash
git checkout release/cicai2026
```

---

## Relationship to the Upstream Project

This implementation extends:

```text
Repository: https://github.com/William-wAng618/roboticAttack
Upstream commit: b8acbc7cbb80544370e73a638d8a663b880268ce
```

The upstream project provides:

- UADA and UPA adversarial patch attacks;
- OpenVLA model integration;
- LIBERO evaluation infrastructure;
- adversarial patch generation and placement;
- robotic rollout and success-rate evaluation.

This repository adds:

- the GEAR inference-time defense pipeline;
- geometry-guided residual attention transformation;
- robot arm and gripper projection;
- suspicious-region localization;
- localized visual purification;
- action, attention, and mask recovery diagnostics;
- runtime instrumentation;
- GRT ablation;
- JPEG compression and Gaussian blur baselines.

The original upstream documentation and historical experimental notes are
retained in:

```text
README_LEGACY.md
```

---

## Main Evaluation Entry Point

The main LIBERO evaluation program is:

```text
experiments/robot/libero/run_libero_eval_args_geo_batch.py
```

This entry point supports:

- clean evaluation;
- adversarial patch evaluation;
- GEAR defense;
- GEAR without GRT;
- defense on clean inputs;
- JPEG and Gaussian blur baselines;
- action and attention diagnostics;
- runtime measurement;
- task-specific and suite-level rollouts.

---

## GEAR Implementation

The main defense implementation is located under:

```text
evaluation_tool/defense/
```

The directory contains modules for:

- attention extraction and stabilization;
- adversarial-region localization;
- robot arm geometry projection;
- gripper geometry projection;
- geometry-residual transformation;
- candidate-region selection;
- image purification;
- mask and recovery metrics;
- runtime and debug logging.

The final paper configuration uses the automatic localization mode:

```text
--defense_mode auto
```

and disables model fine-tuning or policy modification.

> Some optional and legacy experimental modules remain in this directory but
> were not enabled in the final paper experiments. The exact paper
> configuration is documented in `../docs/reproduction.md`.

---

## Evaluation Modes

The evaluation modes used in the paper can be represented as follows.

| Mode | Patch | Defense | GRT | Image baseline |
|---|---:|---:|---:|---|
| `clean` | No | No | — | None |
| `attack` | Yes | No | — | None |
| `defense_on_clean` | No | Yes | Yes | None |
| `gear` | Yes | Yes | Yes | None |
| `gear_no_grt` | Yes | Yes | No | None |
| `jpeg` | Yes | No | — | JPEG |
| `gaussian_blur` | Yes | No | — | Gaussian blur |

### Clean evaluation

```text
--use_patch False
--defense_enabled False
```

### Attack-only evaluation

```text
--use_patch True
--defense_enabled False
```

### GEAR on clean inputs

```text
--use_patch False
--defense_enabled True
--defense_use_residual_candidate_grid True
```

### Full GEAR

```text
--use_patch True
--defense_enabled True
--defense_mode auto
--defense_use_residual_candidate_grid True
```

### GEAR without GRT

```text
--use_patch True
--defense_enabled True
--defense_mode auto
--defense_use_residual_candidate_grid False
```

When GRT is disabled, the localizer directly uses the stabilized attention map
\(Z_t\). Full GEAR instead uses the geometry-residual evidence
\(Z_t^{\mathrm{res}}\).

### JPEG baseline

```text
--use_patch True
--defense_enabled False
--cv_baseline_enabled True
--cv_baseline_strategy jpeg
```

### Gaussian blur baseline

```text
--use_patch True
--defense_enabled False
--cv_baseline_enabled True
--cv_baseline_strategy gaussian_blur
```

The JPEG and Gaussian blur baselines are applied after adversarial patch
insertion. They use the same adversarial patch and patch position as the
corresponding attack-only evaluation.

---

## LIBERO Suites

| Display name | `task_suite_name` | OpenVLA checkpoint | Patch position |
|---|---|---|---:|
| Object | `libero_object` | `openvla/openvla-7b-finetuned-libero-object` | `(0, 174)` |
| Spatial | `libero_spatial` | `openvla/openvla-7b-finetuned-libero-spatial` | `(174, 174)` |
| Goal | `libero_goal` | `openvla/openvla-7b-finetuned-libero-goal` | `(0, 174)` |
| Long | `libero_10` | `openvla/openvla-7b-finetuned-libero-10` | `(0, 174)` |

The main task-success experiments use:

```text
10 tasks per suite
50 rollouts per task
```

Diagnostic experiments use a separate rollout configuration and should not be
used to replace the main task-success results.

---

## Adversarial Patches

The adversarial patch files are located under:

```text
adversarial_patches/simulation/untargeted/
```

### UADA

```text
adversarial_patches/simulation/untargeted/
└── UADA-dof1-b55bb4ee-f3df-4410-b7ea-fcfe68ae4132/
    └── patch.pt
```

### UPA

```text
adversarial_patches/simulation/untargeted/
└── UPA-dof1~3-b9972dd8-8c22-4a2b-923a-92ee14d96e74/
    └── patch.pt
```

UADA and UPA use the same suite-specific patch positions listed above.

---

## Paper Configuration

The final Full GEAR experiments use the following key settings:

```text
--defense_enabled True
--defense_mode auto
--defense_purifier_strategy blend_gray

--defense_localizer_min_area 0.002
--defense_localizer_max_area 0.25

--defense_use_residual_candidate_grid True
--defense_geometry_residual_gamma 1.0
--defense_geometry_residual_normalize_mode original_sum

--defense_geometry_guard_weight_arm 0.6
--defense_geometry_guard_weight_gripper 0.7
--defense_geometry_core_weight_arm 1.0
--defense_geometry_core_weight_gripper 1.0

--defense_verifier_enabled False
--defense_gripper_prior_enabled True
--defense_arm_skeleton_enabled True
--defense_arm_body_names "base,link1,link2,link3,link4,link5"

--defense_near_task_tau 0.08
--defense_allow_near_task_patch False
```

For the GRT ablation, only the following setting is changed:

```text
--defense_use_residual_candidate_grid False
```

All other defense settings should remain unchanged.

The complete commands for the four suites are maintained in:

```text
../docs/reproduction.md
```

---

## Environment Variables

Avoid placing machine-specific absolute paths directly in public commands.

Recommended environment variables:

```bash
export ROBOTIC_ATTACK_MODEL_ROOT=/path/to/models
export LIBERO_DATASET_PATH=/path/to/datasets
export GEAR_OUTPUT_ROOT=/path/to/results
```

The evaluation program can then access model, dataset, and output locations
without relying on the original experimental server paths.

The exact environment snapshot will be provided through:

```text
../environment.yml
../requirements-lock.txt
```

---

## Direct Evaluation Example

The following example evaluates one Object-suite task with Full GEAR.

```bash
python experiments/robot/libero/run_libero_eval_args_geo_batch.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --single_task_id 0 \
  --num_trials_per_task 1 \
  --center_crop True \
  --use_patch True \
  --patchroot adversarial_patches/simulation/untargeted/UADA-dof1-b55bb4ee-f3df-4410-b7ea-fcfe68ae4132/patch.pt \
  --x 0 \
  --y 174 \
  --angle 0 \
  --shx 0 \
  --shy 0 \
  --defense_enabled True \
  --defense_mode auto \
  --defense_purifier_strategy blend_gray \
  --defense_localizer_min_area 0.002 \
  --defense_localizer_max_area 0.25 \
  --defense_use_residual_candidate_grid True \
  --defense_geometry_residual_gamma 1.0 \
  --defense_geometry_residual_normalize_mode original_sum \
  --defense_geometry_guard_weight_arm 0.6 \
  --defense_geometry_guard_weight_gripper 0.7 \
  --defense_geometry_core_weight_arm 1.0 \
  --defense_geometry_core_weight_gripper 1.0 \
  --defense_verifier_enabled False \
  --defense_gripper_prior_enabled True \
  --defense_arm_skeleton_enabled True \
  --defense_arm_body_names "base,link1,link2,link3,link4,link5" \
  --defense_near_task_tau 0.08 \
  --defense_allow_near_task_patch False \
  --defense_viz False \
  --defense_debug False \
  --use_wandb False \
  --cudaid 0 \
  --local_log_dir ../outputs/smoke_test/object \
  --run_id_note object_uada_gear_smoke
```

This command uses only one rollout and is intended as a smoke test, not as a
replacement for the paper evaluation.

---

## Diagnostics

The evaluation program can record:

- mask-patch IoU;
- patch recall and precision;
- Patch Attention Mass;
- attention suppression;
- clean, adversarial, and defended actions;
- action L2 deviation;
- NAR-L2;
- task and episode success.

Diagnostic recording requires:

```text
--metrics_enabled True
--metrics_action_recovery True
--metrics_attention_recovery True
--metrics_mask_recovery True
```

Released task-level diagnostic data are located in:

```text
../results/diagnostics/raw/
```

Reproduce the released summary from the repository root:

```bash
python scripts/summarize_diagnostics.py \
  --input_root results/diagnostics/raw \
  --output_dir results/diagnostics
```

The released diagnostic data contain:

```text
4 suites
10 tasks per suite
15 diagnostic rollouts per task
600 diagnostic episodes in total
```

---

## Runtime Evaluation

Runtime instrumentation is supported through:

```text
--runtime_metrics_enabled True
--runtime_warmup_steps 10
--runtime_save_jsonl True
--runtime_jsonl_name runtime_metrics.jsonl
--runtime_summary_name runtime_summary.json
```

Runtime logs can be summarized with:

```bash
python scripts/summarize_runtime.py \
  --input_glob "/path/to/runtime_results/*/*/runtime_metrics.jsonl" \
  --output_csv "/path/to/runtime_results/runtime_summary_all.csv"
```

The historical runtime experiment compares:

- clean OpenVLA inference;
- OpenVLA inference with GEAR enabled.

The exact runtime configuration used in the paper will be retained separately
from the main task-success configuration.

---

## Output Files

Depending on the enabled options, evaluation outputs may include:

```text
local logs
rollout videos
episode summaries
task summaries
frame-level metric JSONL files
runtime metric JSONL files
debug visualizations
```

Large generated outputs should not be committed to Git.

Recommended ignored directories include:

```text
logs/
local_logs/
outputs/
results/generated/
rollouts/
wandb/
```

Released compact result summaries are maintained under the repository-level:

```text
../results/
```

---

## Reproduction Documentation

See the following repository-level documents:

- [Installation](../docs/installation.md)
- [Paper reproduction](../docs/reproduction.md)
- [Diagnostic results](../results/diagnostics/README.md)
- [Third-party notices](../THIRD_PARTY_NOTICES.md)

---

## Legacy Documentation

The original upstream README and historical experimental commands are retained
in:

```text
README_LEGACY.md
```

These notes may contain:

- server-specific absolute paths;
- historical attack-generation commands;
- abandoned experimental settings;
- settings that do not correspond to the final GEAR paper.

For the paper configuration, use this README and the repository-level
reproduction documentation instead.

---

## License and Attribution

The original upstream license is retained in:

```text
LICENSE
```

This directory contains code derived from the upstream roboticAttack project.
The repository-level `THIRD_PARTY_NOTICES.md` documents upstream sources and
licenses.

The license for the newly added GEAR implementation will be finalized at the
repository root after confirmation by the authors.
