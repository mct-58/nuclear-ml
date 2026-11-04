# Nuclear x ML Project

Learning about Nuclear physics and Machine Learning from scratch simultaneously and then combining them in a final project.

## Overview

4 stages of Physics simulations and 4 stages of Machine Learning progression done in parallel to finally merge them in a neural surrogate for reactor criticality. Everything written by hand from scratch with only NumPY and Matplotlib (plotting was assisted for time efficiency). The point was to understand the systems and tools that are widely used without second thought in real reactor design - like the Monte Carlo methods.

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