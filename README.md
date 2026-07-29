<div align="center">

# GEAR

### Geometry-Guided Evidence for Action Recovery in  
### Inference-Time Defense of Vision-Language-Action Models

**CICAI 2026**

Siyuan Zhu · Jiawei Tu · Jianwei Hou · Zifeng Kang

<br>

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-ee4c2c)
![OpenVLA](https://img.shields.io/badge/VLA-OpenVLA--7B-6f42c1)
![Benchmark](https://img.shields.io/badge/Benchmark-LIBERO-2ea44f)
![Conference](https://img.shields.io/badge/CICAI-2026-orange)
![Status](https://img.shields.io/badge/Status-Camera--Ready-success)

</div>

---

## Overview

Vision-Language-Action models directly convert visual observations and
language instructions into robotic actions. This creates a critical security
risk: a localized adversarial patch in the camera observation can corrupt the
visual evidence used by the policy and consequently mislead robotic actions.

**GEAR** is a plug-in, input-side defense against non-occluding adversarial
patch attacks on frozen VLA policies. It does not fine-tune or modify the
deployed policy.

GEAR projects robot geometry into the policy image space, discounts
robot-attributable attention, and uses the remaining residual evidence to
localize suspicious regions for targeted input purification.

### Key properties

- **Inference-time defense:** no VLA model retraining or fine-tuning.
- **Policy-agnostic integration:** the deployed action policy remains frozen.
- **Geometry-guided localization:** robot arm and gripper geometry are
  projected into the image space.
- **Localized purification:** only suspicious image regions are processed.
- **Action-level evaluation:** recovery is measured beyond task success rate.

---

## Method

The GEAR inference pipeline consists of four main stages:

1. Extract and temporally stabilize visual attention from the frozen VLA model.
2. Project robot arm and gripper geometry into the policy image space.
3. Apply the Geometry-Residual Transformation to obtain residual evidence
   \(Z_t^{\mathrm{res}}\).
4. Localize and purify suspicious regions before the final action prediction.

<!-- Replace this placeholder after adding the framework figure. -->

<p align="center">
  <i>Framework figure will be added under <code>assets/framework.png</code>.</i>
</p>

<!--
<p align="center">
  <img src="assets/framework.png" width="92%" alt="GEAR framework">
</p>
-->

---

## Main Results

We evaluate GEAR on **OpenVLA-7B** across four LIBERO suites under the
UADA adversarial patch attack.

| Method | Average Task Success Rate | Attack Recovery Rate |
|---|---:|---:|
| Attack-only | 2.0% | 0.0% |
| GEAR without GRT | 59.1% | 86.0% |
| **GEAR** | **62.2%** | **89.5%** |

GEAR additionally recovers up to **71.8%** of the attack-induced action
deviation.

> The diagnostic experiments and the main task-success experiments use
> different rollout counts. Diagnostic success rates should therefore not be
> interpreted as replacements for the main paper results.

---

## Diagnostic Results

The repository includes released task-level diagnostic summaries for:

- Patch Attention Mass;
- Mask-patch IoU;
- Patch recall;
- Action L2 deviation;
- Normalized Action Recovery;
- Task success during diagnostic rollouts.

The diagnostic data contain:

- four LIBERO suites;
- ten tasks per suite;
- fifteen diagnostic rollouts per task;
- six hundred diagnostic episodes in total.

Reproduce the released diagnostic summary with:

```bash
python scripts/summarize_diagnostics.py \
  --input_root results/diagnostics/raw \
  --output_dir results/diagnostics
```

Generated files:

```text
results/diagnostics/
├── suite_diagnostics.csv
├── suite_diagnostics.json
├── task_diagnostics.csv
└── raw/
```

---

## Repository Status

The paper experiments correspond to the following code version:

```text
Branch: release/cicai2026
Experimental commit: d1dc5a88e27d8293b7add03561c12b8519bc7774
```

The `main` branch contains an earlier development path and does not exactly
match the final mechanism reported in the paper. For paper reproduction,
always use:

```bash
git checkout release/cicai2026
```

A stable release tag will be created after the documentation and reproduction
scripts are finalized.

---

## Repository Structure

```text
RoboticAttack_Defense/
├── README.md
├── THIRD_PARTY_NOTICES.md
├── LICENSE                         # To be finalized
├── CITATION.cff                    # To be added after publication
│
├── roboticAttack/
│   ├── README.md                   # Implementation documentation
│   ├── README_LEGACY.md            # Historical upstream notes
│   ├── adversarial_patches/
│   ├── evaluation_tool/
│   │   └── defense/                # GEAR defense modules
│   └── experiments/
│       └── robot/libero/
│
├── LIBERO/                         # Unmodified LIBERO benchmark
│
├── scripts/
│   ├── summarize_diagnostics.py
│   └── ...                         # Reproduction scripts
│
├── results/
│   └── diagnostics/
│
├── docs/
│   ├── installation.md
│   └── reproduction.md
│
├── configs/
│   └── ...                         # Experiment configurations
│
└── assets/
    ├── framework.png
    ├── results.png
    └── demo.gif
```

---

## Supported Experimental Settings

### LIBERO suites

| Display name | Suite argument | OpenVLA checkpoint | Patch position |
|---|---|---|---:|
| Object | `libero_object` | `openvla/openvla-7b-finetuned-libero-object` | `(0, 174)` |
| Spatial | `libero_spatial` | `openvla/openvla-7b-finetuned-libero-spatial` | `(174, 174)` |
| Goal | `libero_goal` | `openvla/openvla-7b-finetuned-libero-goal` | `(0, 174)` |
| Long | `libero_10` | `openvla/openvla-7b-finetuned-libero-10` | `(0, 174)` |

### Evaluation modes

| Mode | Adversarial patch | GEAR | GRT |
|---|---:|---:|---:|
| `clean` | No | No | — |
| `attack` | Yes | No | — |
| `defense_on_clean` | No | Yes | Yes |
| `gear` | Yes | Yes | Yes |
| `gear_no_grt` | Yes | Yes | No |
| `jpeg` | Yes | No | — |
| `gaussian_blur` | Yes | No | — |

For the GRT ablation, the stabilized attention map \(Z_t\) is directly used
for localization:

```bash
--defense_use_residual_candidate_grid False
```

For full GEAR, the geometry-residual attention map
\(Z_t^{\mathrm{res}}\) is used:

```bash
--defense_use_residual_candidate_grid True
```

---

## Adversarial Patches

The released evaluation supports UADA and UPA patches.

```text
roboticAttack/adversarial_patches/simulation/untargeted/
├── UADA-dof1-b55bb4ee-f3df-4410-b7ea-fcfe68ae4132/
│   └── patch.pt
└── UPA-dof1~3-b9972dd8-8c22-4a2b-923a-92ee14d96e74/
    └── patch.pt
```

UADA and UPA use the same suite-specific patch positions listed above.

---

## Installation

Detailed environment instructions will be maintained in:

```text
docs/installation.md
```

The main dependencies include:

- Python;
- PyTorch and CUDA;
- OpenVLA;
- LIBERO;
- MuJoCo;
- robosuite;
- Transformers;
- OpenCV.

The exact environment snapshot will be released through:

```text
environment.yml
requirements-lock.txt
```

---

## Reproduction

Detailed commands will be maintained in:

```text
docs/reproduction.md
```

The final reproduction interface will support commands of the following form:

```bash
bash scripts/run_suite.sh \
  --suite object \
  --attack uada \
  --mode gear \
  --trials 50 \
  --cuda 0
```

Supported arguments will include:

```text
suite:
  object
  spatial
  goal
  long

attack:
  uada
  upa
  none

mode:
  clean
  attack
  defense_on_clean
  gear
  gear_no_grt
  jpeg
  gaussian_blur
```

Until the unified runner is finalized, the exact historical commands are
retained in the implementation documentation.

---

## Relationship to Upstream Projects

This repository is built upon the following open-source projects.

### roboticAttack

Upstream repository:

```text
https://github.com/William-wAng618/roboticAttack
```

Upstream commit:

```text
b8acbc7cbb80544370e73a638d8a663b880268ce
```

The upstream repository provides:

- adversarial patch generation;
- UADA and UPA attack implementations;
- OpenVLA integration;
- LIBERO evaluation infrastructure.

This repository extends it with:

- the GEAR defense pipeline;
- geometry-residual attention transformation;
- robot arm and gripper projection;
- localized purification;
- action, attention, and mask recovery metrics;
- runtime instrumentation;
- GRT ablation;
- JPEG and Gaussian blur baselines.

### LIBERO

This repository includes an unmodified copy of LIBERO as the robotic
manipulation benchmark.

The original LIBERO license and attribution are retained in:

```text
LIBERO/LICENSE
```

### OpenVLA

OpenVLA-7B is used as the victim Vision-Language-Action policy. Model usage
remains subject to the original OpenVLA license and model terms.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for detailed
third-party attribution.

---

## Paper

**GEAR: Geometry-Guided Evidence for Action Recovery in Inference-Time
Defense of Vision-Language-Action Models**

Accepted by **CICAI 2026**.

The official publication link and DOI will be added after publication.

---

## Citation

The formal BibTeX entry will be added after the proceedings are published.

Temporary citation information:

```bibtex
@inproceedings{zhu2026gear,
  title     = {{GEAR}: Geometry-Guided Evidence for Action Recovery in
               Inference-Time Defense of Vision-Language-Action Models},
  author    = {Zhu, Siyuan and Tu, Jiawei and Hou, Jianwei and Kang, Zifeng},
  booktitle = {Chinese Conference on Artificial Intelligence},
  year      = {2026}
}
```

---

## Acknowledgments

This project builds upon the open-source implementations of roboticAttack,
OpenVLA, and LIBERO. We thank their authors for releasing the corresponding
code, models, and benchmark environments.

Additional acknowledgments and funding information will be updated according
to the final paper.

---

## License

Third-party components retain their original licenses:

- `roboticAttack/LICENSE`;
- `LIBERO/LICENSE`.

The license for the original GEAR implementation will be added after
confirmation by the authors.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details.
