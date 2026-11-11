"""Run the two-layer bond GNN in NumPy and keep its steps for the viewer."""
import json
import numpy as np
from rdkit import Chem
from gnn.molecule import smiles_to_graph, normalize


def load_bundle(path):
    # model.npz is what the notebook exported: the trained weights plus the whole
    # dataset with the score PyTorch gave every molecule
    with np.load(path, allow_pickle=False) as saved:
        weights = {name: saved[name] for name in saved.files}
    metadata = json.loads(str(weights["metadata"]))
    # we refuse any other architecture here, because everything below assumes these three
    # things. a mean-pooling or one-layer file would still load and would quietly draw a
    # picture that does not match its own weights
    if (metadata.get("readout") != "sum" or metadata.get("use_bonds") is not True
            or metadata.get("message_layers") != 2):
        raise ValueError("This viewer needs the two-layer bond model with sum pooling")
    return weights


def message_layer(values, adjacency, weights, layer, bond_types):
    """Self and four bond transforms share the same degree normalization."""
    suffix = str(layer)
    # code 5 is the atom's own vector, so this is the self-loop part of the sum
    transformed = values @ weights["w" + suffix].T + weights["b" + suffix]
    messages = (adjacency * (bond_types == 5)) @ transformed
    # then one pass per bond type, each with its own weights. the codes are 1 single,
    # 2 double, 3 triple, 4 aromatic, and multiplying by the mask keeps only the bonds
    # of that type, which is the whole reason a double bond sends something different
    for kind in range(4):
        transformed = values @ weights["wb" + suffix][kind].T + weights["bb" + suffix][kind]
        messages += (adjacency * (bond_types == kind + 1)) @ transformed
    # relu, same as the notebook
    return np.maximum(messages, 0)


def trace_molecule(smiles, weights):
    # this is the notebook's forward pass again, but we keep every intermediate array,
    # because the animation has to show what each atom was holding at each step
    graph = smiles_to_graph(str(smiles))
    if graph is None:
        raise ValueError("Cannot read molecule: " + str(smiles))
    bond_types, x = graph
    # we take the diagonal back out, so adjacency means neighbors only. the atom's own
    # vector is already handled by the code 5 branch above
    adjacency = (bond_types > 0).astype(float) - np.eye(len(x))
    normalized = normalize(adjacency)
    # h1 is after one round, h2 after two. two rounds means every atom has seen
    # everything within two bonds of itself
    h1 = message_layer(x, normalized, weights, 1, bond_types)
    h2 = message_layer(h1, normalized, weights, 2, bond_types)
    # sum pooling and not mean, so one active group does not get diluted in a big molecule
    pooled = h2.sum(axis=0)
    logit = float(pooled @ weights["wout"].reshape(-1) + weights["bout"].item())
    # sigmoid written the stable way, because exp of a large positive logit overflows
    exp = np.exp(-abs(logit))
    score = 1 / (1 + exp) if logit >= 0 else exp / (1 + exp)
    return dict(mol=Chem.MolFromSmiles(str(smiles)), features=x, adjacency=adjacency,
                normalized=normalized, bond_types=bond_types, hidden_states=[h1, h2],
                h1=h1, h2=h2, pooled=pooled, logit=logit, probability=float(score))


def validate_bundle(weights, tolerance=2e-5):
    # we rescore all 3839 molecules and compare against what PyTorch got in the notebook,
    # because this NumPy version is a second implementation and it could disagree without
    # anything looking wrong on screen
    scores = [trace_molecule(s, weights)["probability"] for s in weights["smiles"]]
    error = float(np.max(np.abs(scores - weights["reference_probabilities"])))
    if not np.isfinite(error) or error > tolerance:
        raise ValueError("Viewer and PyTorch predictions differ: " + str(error))
    # run.py prints this at startup. it comes out at 4.5e-07, which is float32 rounding
    return error
