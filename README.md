# pH-integration
This repository stores and shows scripts to integrate experimental data (such as NMR) with data from constant pH (CpH) MD simulations.

The project includes the `ph_refine.py` file (with all the functions required for the integration with experimental data) and some Jupyter notebooks, used either in the implementation process and as tutorials to describe the functionalities of `ph_refine.py`.

The main concepts and the loss function here implemented are described in the documentation.

---

## 📁 Repository structure

```
pH-integration/
├── Simulation-data/
│ ├── molname/ # for each molecular system, a directory with the name of the molecular system `molname` (such as `A3mer` or `A5mer`) contains the data from experiments and MD simulations for that system
│ │ ├── molname_pH%s.occ  # 
│ │ └── COLVAR_REWEIGHT_%s  # 
        HILLS_pH%s.gz  # 
│       TOTVAR_REWEIGHT_%s  #
|
├── ph_refine.py
│ 
├── ph_refine.ipynb
│ 
├── ph_optimization_draft.ipynb  # 
│ 
├── test.ipynb  # 
│
├── ph_optimization.ipynb  # 
│ 
├── Images/  # 
└── README.md # This file
```

## Project overview

This project integrates experimental data from measures at multiple pH values with CpH-MD simulations.
To augment the statistics, we collect frames at given protonation state from multiple pH values in order to get a canonical ensembles for each protonation state. We also exploit the grand-canonical statistics.

---

## Quick start: minimal analysis (from DATA_structures)

- Load the data with `load_data`
- Evaluate the loss function `ph_loss`
- Minimize it as described in the [`ph_optimization.ipynb`](ph_optimization.ipynb) notebook, by using the function `ph_tilde_loss_and_grad` of [`ph_refine.py`](ph_refine.py).
- Analyse the results of this optimization.

---

## Data description: `Simulation-data/`

This directory contains all you need for the integrative methods described in this repository through the functions in [`ph_refine.py`](ph_refine.py):

- `A3mer_ph03.00.occ`  
  → 

- `COLVAR_REWEIGHT_03.00`  
  → 

- `HILLS_pH03.00.gz`  
  → 

- `TOTVAR_REWEIGHT_03.00`  
  → 

---

## 📦 Dependencies

The main function [`ph_refine.py`](ph_refine.py) requires the following Python libraries:

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

- [Open an issue](https://github.com/IvanGilardoni/MD_analysis_oligomers/issues)
- Contact the author: `igilardo@sissa.it`
