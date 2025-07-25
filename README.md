# Autonomous Radiation Source Localisation and Estimation

This repository implements a simulation-based framework for the autonomous localisation and strength estimation of multiple radioactive sources in a 2D environment. The approach combines particle filtering with deep Q-learning (DQN), using a goal-directed reinforcement learning strategy to actively guide exploration and improve source estimation.

## Core Features

- Particle filter for estimating number, location, and intensity of sources.
- Goal-directed deep Q-learning (DQN) to actively navigate the agent toward informative areas in the environment.
- Simulation outputs include:
  - Radiation heatmaps
  - Agent movement visualisation
  - Estimation error plots
  - Source strength and count estimation graphs

## Repository Structure

- `radiation_STE_discrete.py` – Main simulation script using goal-directed DQN and particle filtering.
- `particle_filter.py` – Particle filter implementation (inspired by Roger R. Labbe Jr.'s Kalman and Bayesian Filters in Python).
- `radiation_discrete.py` – Environment and agent/source class definitions.

## Running the Simulation

### Requirements

- Python 3.9+
- PyTorch
- NumPy
- Matplotlib
- Statsmodels

### To Run

Execute the main simulation file:

```bash
python radiation_STE_discrete.py
```

Training progress and plots will be displayed live (if `displayPlot` is set to `True`) and results will be saved in the `multi_source_results/` directory.

### Notes

- The code auto-detects GPU or MPS acceleration (Mac) if available.
- Simulations may take several hours depending on the hardware. A Colab-compatible version can be adapted.

## Output

All key performance metrics, including source localisation error, source count estimates, and intensity error are visualised and exported as:

- `distance_plot.pkl` / `dist.png`
- `length_plot.pkl`
- `est_strengths.png`
- `num_sources.png`
- `estimation_error.png`

## Future Work

The simulation can be extended to:
- Support adaptive ε-decay strategies.
- Improve parameter tuning via Bayesian optimisation.
- Introduce transfer learning across different environments.
- Evaluate robustness under randomised agent initialisation.
