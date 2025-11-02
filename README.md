# Me-DrugBAN: Bilinear Attention Networks for Drug-Target Interaction

This repository is an annotated **copy and educational reimplementation of the original [DrugBAN](https://github.com/peizhenbai/DrugBAN)** project. It is tailored for learning and experimentation in a reproducible, modular way, and runs smoothly in WSL (Windows Subsystem for Linux).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Environment Setup (with WSL support)](#environment-setup-with-wsl-support)
- [Dataset Preparation](#dataset-preparation)
- [Project Structure](#project-structure)
- [Training Pipeline](#training-pipeline)
- [Testing & Evaluation](#testing--evaluation)
- [Reproducibility Checklist](#reproducibility-checklist)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This repository contains a PyTorch re-type and annotation of DrugBAN: a bilinear attention network (BAN) for drug–target interaction (DTI) prediction by combining molecular graph features (for drugs) and protein sequence features, fused by a bilinear attention mechanism. This implementation is modular, allowing easy extension and reproducibility.We copy and reimplement the original DrugBAN pipeline.

---

## Environment Setup (with WSL support)

1. **Clone the repository:**
   ```sh
   git clone https://github.com/HussenYesufAlli/Me-DrugBAN.git
   cd Me-DrugBAN
   ```

2. **If using WSL:**  
   Ensure you have conda (Anaconda/Miniconda) or mamba installed in your WSL environment. Follow [this guide](https://docs.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers) if needed.

3. **Create and activate the conda environment:**
   ```sh
   conda env create -f environment.yml
   conda activate me-drugban
   ```
   > _This installs Python 3.8, PyTorch, DGL, RDKit, dgllife, and all other dependencies._

---

## Dataset Preparation

1. **Obtain the datasets** (BindingDB, BioSNAP, Human, etc.) in the following format:
   ```csv
   SMILES,Protein,Y
   CC(C)CC1=CC=C(C=C1)C(C)C(=O)O,MEGTVK...,1
   ```

2. **Place datasets under the `data/` directory (note: not `datasets/`):**
   ```
   data/
     bindingdb/
       random/
         train.csv
         val.csv
         test.csv
       cluster/
         source_train.csv
         target_train.csv
         target_test.csv
       full.csv
     bindingdb_sample/
       train.csv
       val.csv
       test.csv
     biosnap/
       ...
     human/
       ...
   ```
   > _This matches the structure in the original DrugBAN but uses `data/` as the top-level directory, as in this repo._

---

## Project Structure

```
Me-DrugBAN/
│
├── data/                  # All dataset CSVs (see above; was 'datasets/' in original DrugBAN)
├── scripts/               # Entrypoint scripts (train.py, etc.)
├── src/
│   └── me_drugban/
│       ├── data_loader/
│       ├── gnn_backbone/
│       ├── protein_encoder/
│       ├── ban_module/
│       ├── train_loop/
│       └── model.py
├── tests/
├── environment.yml
├── README.md
├── pyproject.toml, setup.py
└── ...
```

---

## Training Pipeline

1. **Edit your training script (e.g. `scripts/train.py`):**
   ```python
   train_path = "data/bindingdb/random/train.csv"
   val_path   = "data/bindingdb/random/val.csv"
   ```

2. **Run training:**
   ```sh
   python scripts/train.py
   ```
   - This will load real data, initialize the model, and train for the specified epochs.
   - Checkpoints and logs will be saved (e.g., `best_drugban.pt`, `training_history.pkl`).

---

## Testing & Evaluation

- For evaluation on the test set, use or adapt `scripts/test_model_forward.py`:
   ```python
   test_path = "data/bindingdb/random/test.csv"
   # ... load best_drugban.pt and evaluate
   ```

---

## Reproducibility Checklist

- [x] Use the provided `environment.yml` to recreate the environment (works in WSL).
- [x] Store and document dataset splits and preprocessing under `data/`.
- [x] Save the exact config/hyperparameters for each run (in configs/ or as script args).
- [x] Save random seed and environment version (`conda list > packages.txt`).
- [x] Record hardware (GPU/CPU, RAM) for each experiment.
- [x] Save trained models and logs.

---

## Citation

If you use this code, please cite:
> Peizhen Bai et al., Interpretable bilinear attention network with domain adaptation improves drug-target prediction. Nature Machine Intelligence (2023). DOI: 10.1038/s42256-022-00605-1

---

## Acknowledgements

- Original [DrugBAN repository](https://github.com/peizhenbai/DrugBAN) (this is a copy and learning reimplementation)
- DGL, dgllife, RDKit, and PyTorch open-source communities

---

_For questions or contributions, please open an issue or pull request._