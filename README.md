# Molecule-Informed GNN

A graph neural network that predicts whether a molecule inhibits *E. coli* growth, and a pygame
walkthrough of what the trained model does to one molecule, one message-passing round at a time.

```bash
pip install -r requirements.txt
python run.py
```

Space steps through the molecule, message round 1, message round 2, and the prediction. `R` resets,
the arrows change molecule, clicking an atom moves the focus, `Esc` quits.

The ring around each atom shows which atoms its vector has absorbed so far, so after two rounds the
rings are the two-hop neighborhood the prediction is built from. Every number on screen is read
from the trained weights in `models/model.npz`.

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
| `originals/` | Jane's code, unchanged |

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

Calling every molecule inactive already scores 72.2% here while finding nothing, so accuracy alone
is misleading. The stopping epoch (133) and the decision threshold (0.688) were chosen on
validation only. The test split was reused across several rounds of exploration, so these numbers
describe this fixed split rather than external validation.

`gnn/predict.py` is a second implementation of the forward pass, in NumPy. It rescores all 3,839
molecules at startup and compares them against PyTorch. The largest difference is 4.5e-07.

## Contributions

**Jane** wrote the baseline, kept unchanged in `originals/`: reading the CSV, `atom_features`,
`smiles_to_graph`, the dataset summary, the pygame renderer with its RDKit coordinates, element
colors, parallel lines for double and triple bonds, numbered atom circles and event loop, and the
first PyTorch GNN with its normalized adjacency, two message-passing layers, linear output, and
Adam training loop. Her `atom_features` and her drawing code are reused character for character,
and the class name and layer shapes in `train.ipynb` are still hers.

**Carmen** applied that baseline to one question. The dataset: 3,839 ChEMBL molecules with a
measured *E. coli* MIC, units filtered, repeat measurements merged, conflicting records dropped,
in place of a 217-row file whose labels are not E. coli MIC. Inside the layer: five relations with
their own weights instead of one binary adjacency matrix, 5,185 parameters to 25,665, which moved
accuracy from 69.1% to 76.6% and precision from 46.0% to 57.7% while recall fell from 61.7% to
59.2% and ROC-AUC from 0.716 to 0.687. Not every metric improved. Sum pooling instead of mean. A
scaffold split with a validation set for the stopping epoch and the threshold, reported as the full
metric set against the 72.2% majority baseline. And the walkthrough: a NumPy forward pass that
keeps every layer, message packets carrying the real vectors along the bonds, and rings that show
the receptive field growing round by round.
