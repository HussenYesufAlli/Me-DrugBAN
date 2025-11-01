# ME-drugBAN
Interpretable bilinear attention network with domain adaptation for drug-target interaction prediction — local copy and notes for my ME-drugBAN project.

## Summary
This repository is a local copy and working space for DrugBAN, a bilinear attention network that models drug–target interactions using 2D drug graphs and protein sequences, with optional adversarial domain adaptation to improve out-of-distribution performance. See: [Nature Machine Intelligence paper](https://doi.org/10.1038/s42256-022-00605-1).

## Quick start (demo)
1. Create and activate a conda environment:
```bash
conda create -n me-drugban python=3.8 -y
conda activate me-drugban
```
2. Install core dependencies (adjust versions as needed):
```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
conda install -c dglteam dgl-cuda10.2==0.7.1
conda install -c conda-forge rdkit==2021.03.2
pip install dgllife==0.2.8 scikit-learn yacs prettytable
```
3. Prepare datasets:
- Place datasets under `datasets/` (see datasets/README.md).
4. Run a demo training (example):
```bash
# If the repo uses a train script named `train.py` and config files in configs/
python train.py --config configs/DrugBAN_Demo.yaml
```
(If the real entrypoint is different, replace `train.py` with your actual script name.)

## System requirements
- Python 3.8
- PyTorch >= 1.7.1
- DGL, dgllife
- RDKit for SMILES → molecule conversions (install via conda-forge)
- Not strictly required: GPU (faster), but CPU should work for small demo runs.

## Project structure (edit to match your local tree)
- configs/ — YAML configs for experiments
- datasets/ — sample and full datasets (train/val/test CSVs)
- models/ or src/ — model implementations (BAN, domain adaptation modules)
- train.py / main.py — training and evaluation entrypoint(s)
- drugban_demo.ipynb — Colab demo notebook
- result/ — output directory for checkpoints, logs and metrics

## Configuration & running experiments
- Edit or copy configs/*.yaml to change hyperparameters (batch size, lr, DA settings).
- Example: to run non-domain-adaptive training:
```bash
python train.py --config configs/DrugBAN_Non_DA.yaml
```
- To run domain-adaptive training:
```bash
python train.py --config configs/DrugBAN_DA.yaml
```
(Confirm the exact CLI arguments in your repo; search for argparse or config loader to find the real flags.)

## Datasets
- Original datasets come from BindingDB, BioSNAP (MolTrans), Human (TransformerCPI). See `datasets/README.md` for references.
- CSV format expected (example header):
```
SMILES,Protein,Y
```
- Place large datasets outside Git if they are big and update .gitignore accordingly.

## Development notes (for my learning)
- I will retype key components (data preprocessing, model implementation, training loop) to fully understand every line.
- Document major changes here with dates and intent.
- Keep experiments reproducible by saving config files and random seeds.

## Running checks & debugging tips
- Search for the training entrypoint:
```bash
grep -R "if __name__ == '__main__'" -n .
```
- Inspect config loader (search for yacs or config file parsing) to see how to pass configs.
- Use small dataset slice (configs/DrugBAN_Demo.yaml) for quick debugging.

## Citation
If you use this work, please cite:
Peizhen Bai et al., Interpretable bilinear attention network with domain adaptation improves drug-target prediction. Nature Machine Intelligence (2022). DOI: 10.1038/s42256-022-00605-1

## License
This project is under the MIT License (adjust if you changed license).
