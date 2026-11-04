# Nuclear x ML Project

Learning nuclear physics simulation and machine learning from scratch at the same time, then combining them in a final project.

## Overview

Four stages of physics simulation and four stages of machine learning, built in parallel, which merge in stage 5 into a neural surrogate model for reactor criticality. Everything is written from scratch using only NumPy and Matplotlib, with no ML frameworks (plotting code was AI-assisted to save time). The point is to understand the methods and tools that are used without a second thought in real reactor design, such as Monte Carlo methods.

## Progress

| Stage | Physics | ML | Status |
|---|---|---|---|
| 1 | Charged particle in a B field | Linear regression | Done |
| 2 | Decay chains, adaptive timestep | Neural network, backprop | Done |
| 3 | Neutron random walk | Monte Carlo, minibatching | In progress |
| 4 | Neutron diffusion, k-eff | Autograd, convolutions | Planned |
| 5 | Neural surrogate for k-eff | | Planned |

## Running it

Requires Python 3 and NumPy/Matplotlib only.

```bash
git clone https://github.com/mct-58/nuclear-ml.git
cd nuclear-ml
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python Stage1/particalInMagneticField/main.py
```

Each script runs on its own and displays its plots.

## Development log

Problems encountered, fixes, and validation for each stage are recorded in [LOG.md](LOG.md).