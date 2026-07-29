<div align="center">

# GEAR

### Geometry-Guided Evidence for Action Recovery in  
### Inference-Time Defense of Vision-Language-Action Models

**CICAI 2026**

Siyuan Zhu · Jiawei Tu · Jianwei Hou · Zifeng Kang

[Paper](#paper) ·
[Installation](docs/installation.md) ·
[Reproduction](docs/reproduction.md) ·
[Citation](#citation)

</div>

---

## Overview

GEAR is a plug-in input-side defense against non-occluding adversarial
patch attacks on frozen Vision-Language-Action models.

GEAR projects robot geometry into the policy image space, suppresses
robot-attributable attention, and uses the residual visual evidence to
localize suspicious regions for targeted input purification.

## Main Results

GEAR is evaluated on OpenVLA-7B across four LIBERO suites under UADA
adversarial patch attacks.

| Setting | Average Task Success Rate | ARR |
|---|---:|---:|
| Attack-only | 2.0% | 0.0% |
| GEAR w/o GRT | 59.1% | 86.0% |
| GEAR | **62.2%** | **89.5%** |

GEAR further recovers **71.8%** of the attack-induced action deviation.

## Repository Status

- Official paper branch: `release/cicai2026`
- Final experimental code base:
  `d1dc5a88e27d8293b7add03561c12b8519bc7774`
- Victim model: OpenVLA-7B
- Benchmark: LIBERO
- Attacks: UADA and UPA

## Repository Structure

```text
.
├── roboticAttack/       # Attack framework and GEAR implementation
├── LIBERO/              # Unmodified LIBERO benchmark
├── scripts/             # Experiment and result-summary scripts
├── configs/             # Reproduction configurations
├── results/             # Released diagnostic summaries
├── docs/                # Installation and reproduction guides
└── assets/              # Figures and visual materials
