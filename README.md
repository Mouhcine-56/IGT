# IGT — Initialization, Generation, and Training for Optimal Control and Mean Field Games

[![arXiv](https://img.shields.io/badge/arXiv-2507.15126-b31b1b.svg)](https://arxiv.org/abs/2507.15126)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A neural-network-based framework for solving **high-dimensional optimal control problems** and **first-order Mean Field Games (MFGs)**.

---

## 📌 Overview

Solving Hamilton–Jacobi–Bellman (HJB) equations and MFG systems in high dimensions is notoriously challenging due to the curse of dimensionality. **IGT** tackles this through a three-stage pipeline:

1. **Initialization** — A neural network is trained to approximate the value function by minimizing the HJB residual (using the Deep Galerkin Method).
2. **Generation** — The learned value function provides an initial guess for Pontryagin's two-point boundary value problem (TPBVP), which is then solved to generate high-accuracy supervised data.
3. **Training** — The network is retrained using a hybrid loss that combines the supervised BVP data with the HJB residual penalty, yielding a refined value function.

This process can be iterated, and for **Mean Field Games**, it is combined with the **fictitious play** algorithm to approximate Nash equilibria.

---

## 🗂️ Repository Structure

```
IGT/
├── Optimal_control/             # Deterministic optimal control problems
│   ├── Evaluating_IGT/
│   │   ├── Smooth/              # Smooth Hamiltonian (quadratic control cost)
│   │   └── NON_Smooth/          # Non-smooth Hamiltonian (L1 control cost)
│   └── Obstacle/                # Obstacle avoidance problem
│
└── Mean field games/            # Mean Field Game experiments
    ├── Evaluating IGT-MFG/      # Linear-quadratic MFG (up to 100D)
    └── MFG_Obstacle/            # MFG with obstacle and congestion costs
```

### File layout (common to each experiment)

| File | Description |
|------|-------------|
| `main.py` | Training and evaluation pipeline |
| `model.py` | Neural network architectures (`V_Net`, `G_Net`) |
| `src/solve.py` | HJB solver, TPBVP data generation, generator training |
| `src/problem/prb.py` | Problem definition (Hamiltonian, dynamics, costs, boundary conditions) |

---

## 🚀 Quick Start

### Requirements

- Python ≥ 3.8
- PyTorch
- NumPy, SciPy
- [GeomLoss](https://www.kernel-operations.io/geomloss/) (for MFG experiments)
- [POT](https://pythonot.github.io/) (for Wasserstein distances)

### Running an experiment

```bash
python main.py 
```

Key arguments (vary by experiment):

| Argument | Description |
|----------|-------------|
| `--dim` | State space dimension | 
| `--num_epoch` | Training iterations (HJB) |
| `--num_samples_hjb` | Sample size for HJB loss |
| `--num_samples_bvp` | Number of BVP trajectories |
| `--Max_Round` | IGT refinement rounds |
| `--lr` | Learning rate |

---

## 🧪 Experiments

### Optimal Control

| Experiment | Hamiltonian | Dimensions tested |
|------------|-------------|-------------------|
| **Smooth** | $H(x,p) = -\tfrac{1}{4}\|p\|^2 + p \cdot x$ | 1, 2, 10, 50, 100 |
| **Non-Smooth** | $H(x,p) = \begin{cases} -\tfrac{\|p\|^2}{4} & \|p\| < 2 \\ -\|p\|+1 & \|p\| \geq 2 \end{cases}$ | 1, 2, 10, 50, 100 |
| **Obstacle** | Obstacle avoidance with penalty | 2, 10, 50 |

### Mean Field Games

| Experiment | Description | Dimensions tested |
|------------|-------------|-------------------|
| **Evaluating IGT-MFG** | Linear-quadratic MFG with Gaussian initial distribution | 1, 10, 50, 100 |
| **MFG Obstacle** | MFG with obstacle penalty and congestion cost | 2, 10 |

---

## 📄 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{assouli2025igt,
  title={Initialization-Driven Neural Generation and Training for High-Dimensional Optimal Control and First-Order Mean Field Games},
  author={Assouli, Mouhcine},
  journal={arXiv preprint arXiv:2507.15126},
  year={2025}
}
```

📝 **Paper:** [arXiv:2507.15126](https://arxiv.org/abs/2507.15126)

---

## 📬 Contact

For questions, suggestions, or feedback, feel free to reach out:

**Mouhcine Assouli** — [mouhcine.assouli@unilim.fr](mailto:mouhcine.assouli@unilim.fr)
