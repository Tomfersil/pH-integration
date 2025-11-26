# pH-integration
This repository stores and shows scripts to integrate experimental data (such as NMR) with data from constant pH (CpH) MD simulations.

The project includes the `pH_refine.py` file (with all the functions developed for the integration of CpH MD simulations with experimental data) and some Jupyter notebooks, used either in the implementation process (`ph_refine.ipynb`, `ph_optimization_draft.ipynb`) and as tutorials to describe the functionalities of `pH_refine.py` (`ph_optimization.ipynb`).

The main concepts and the loss function here implemented are described in the documentation `theory.pdf`.

---

## Repository structure

```
pH-integration/
pH-integration/
└── Simulation-data/                    # Input data from constant pH MD simulations (and experiments)
│
├── Images/                              # Figures from notebooks
│
├── theory.pdf                           # Theoretical documentation and derivations
│
├── pH_refine.py                         # Core implementation: pH-refinement routines and utilities
│
├── ph_refine.ipynb                      # Development notebook for pH_refine.py
│
├── ph_optimization_draft.ipynb          # Early notebook for optimization-strategy exploration
│
├── ph_optimization.ipynb                # Final optimization workflow and validation
│
├── test.ipynb                           # Comparison of multiple constant-pH MD simulations
│
├── WHAM.ipynb                           # WHAM analysis to merge multiple constant-pH MD runs into canonical ensembles
│
└── README.md                             # Repository documentation

```

## Project overview

This project integrates experimental data from measures at multiple pH values with CpH-MD simulations.
To augment the statistics, we collect frames at given protonation state from multiple pH values in order to get a canonical ensembles for each protonation state. We also exploit the grand-canonical statistics.

---

## Quick start: minimal analysis (from DATA_structures)

- Load the data as an instance of `PHData` class
- Evaluate the loss function `pH_loss`
- Minimize it as described in the [`ph_optimization.ipynb`](ph_optimization.ipynb) notebook, by using the function `pH_tilde_loss_and_grad` of [`pH_refine.py`](pH_refine.py).
- Analyse the results of this optimization.

---

## Data description: `Simulation-data/`

This directory contains all you need for the integrative methods described in this repository through the functions in [`pH_refine.py`](pH_refine.py), as described in the following.

```

Simulation-data/
  └── <molname>/                                   # One directory per molecular system (e.g., A3mer, A5mer)
      │
      ├── <molname>_pH<pH>.occ                     # Protonation-state occupancies from constant-pH MD
      │
      ├── COLVAR_REWEIGHT_<pH>                     # Reweighted COLVAR file (no metadynamics weights)
      │
      ├── COLVAR_REWEIGHT_<pH>_weighted            # COLVAR with metadynamics weights included
      │                                             # (used to generate TOTVAR_REWEIGHT_<pH>)
      │
      ├── HILLS_pH<pH>.gz                           # Metadynamics HILLS files (if metadynamics-based constant pH MD simulations)
      │
      ├── TOTVAR_REWEIGHT_<pH>                      # Concatenation of weighted COLVAR files, reweighted with the pH factor; used as input of `WHAM.ipynb` notebook
      │
      ├── WHAM_df.txt                               # WHAM detailed output:
      │                                             # columns: weight, n_prot, observables sampling of canonical ensemble at each protonation state
      │
      └── WHAM_pops_<refpH>.txt                     # WHAM-estimated protonation-state populations at reference pH <refpH>
  │
  └── ...                             # Additional molecular systems

```

---

## Dependencies

The main function [`pH_refine.py`](pH_refine.py) requires the following Python libraries:

```
typing
numpy
pandas
jax
scipy
bussilab
MDRefine
```

The notebooks [`ph_refine.ipynb`](ph_refine.ipynb) and [`ph_optimization.ipynb`](ph_optimization.ipynb) requires also the following Python libraries:

```
matplotlib
```

## Scientific context

This project enables systematic integration of experimental data into multiple CpH-MD simulations by taking advantage of the grand-canonical statistics.

---

## Questions or feedback?

Feel free to:

- [Open an issue](https://github.com/Tomfersil/pH-integration/issues)
- Contact the authors: `igilardo@sissa.it`
