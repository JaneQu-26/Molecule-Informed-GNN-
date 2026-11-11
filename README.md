## Molecule-Informed GNN

A graph neural network that predicts whether a molecule inhibits *E. coli* growth, and a pygame
walkthrough of what the trained model does to one molecule, one message-passing round at a time.

```bash
pip install -r requirements.txt
python run.py
```

## Files

| Path | What it is |
|---|---|
| `run.py` | Entry point |
| `visualization/explorer.py` | The four steps, keyboard and mouse, window loop |
| `visualization/drawing.py` | All pygame drawing |
| `gnn/molecule.py` | SMILES to a bond-type adjacency matrix and 14 atom features |
| `gnn/predict.py` | NumPy forward pass that records every layer |
| `train.ipynb` | Colab notebook: load, train, evaluate, export `model.npz` |
| `data/ecoli_mic.csv` | The 3,839 labeled molecules |
| `models/model.npz` | Trained weights, the dataset, and its scores |
| `originals/` | Initial data layer, molecule renderer, and baseline GNN |

## Model

3,839 ChEMBL molecules with a measured MIC against *E. coli* (target CHEMBL354), active below 10 µM.
14 atom features, two message-passing layers, a sum over atoms, one linear output, 25,665 parameters.

Each layer gives every bond type its own weight matrix (single, double, triple, aromatic, and the
atom's own vector), so a double bond does not send the same message as a single bond. Each message
is divided by the square root of both atom degrees. Summing over atoms instead of averaging keeps a
small active group from being diluted in a large molecule.

## Results

Bemis-Murcko scaffold split, so no scaffold appears in two sets: 2,576 train, 832 validation,
431 test.

| Test metric | Value |
|---|---|
| accuracy | 76.6% |
| precision | 57.7% |
| recall | 59.2% |
| ROC-AUC | 0.687 |

Confusion matrix `[[259, 52], [49, 71]]`, rows true inactive/active.


## Contributions

**Jane** wrote the data layer that turns a SMILES string into a graph (`atom_features`, `smiles_to_graph`, CSV loading) and the pygame renderer that draws a molecule on screen (RDKit coordinates, element colors, multi-bond lines, and atom circles). Her baseline code is preserved in `originals/`.

**Both** wrote the first PyTorch GNN with its two message-passing layers, normalized adjacency, linear output, and Adam training loop.

**Carmen** added the domain-specific features to address *E. coli* inhibition: processing the 3,839-molecule ChEMBL dataset, adding bond-type weights and sum pooling, implementing the scaffold train/val/test split, and reporting the full evaluation metrics. She also built the stepped animation of the trained model.
