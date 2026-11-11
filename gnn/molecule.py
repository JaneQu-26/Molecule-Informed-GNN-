"""Jane's atom features with four chemical bond types."""
import numpy as np
from rdkit import Chem

elements = ["C", "N", "O", "S", "F", "Cl", "Br", "P"]
feature_names = elements + ["other element", "degree / 4", "aromatic", "in ring",
                            "formal charge", "hydrogens / 4"]
bond_names = ["single", "double", "triple", "aromatic"]


def atom_features(atom):
    # one-hot for the element, then a few chemical properties
    symbol = atom.GetSymbol()
    onehot = [float(symbol == e) for e in elements] + [float(symbol not in elements)]
    return onehot + [atom.GetDegree() / 4, float(atom.GetIsAromatic()),
                     float(atom.IsInRing()), float(atom.GetFormalCharge()),
                     atom.GetTotalNumHs() / 4]



def smiles_to_graph(smiles):
    # Codes 1-4 are bond types, 5 is self, and 0 means no connection.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
             Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    n = mol.GetNumAtoms()
    a = np.zeros((n, n), dtype=np.uint8)
    np.fill_diagonal(a, 5)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        a[i, j] = a[j, i] = types.index(bond.GetBondType()) + 1
    x = np.array([atom_features(atom) for atom in mol.GetAtoms()])
    return a, x


def normalize(a):
    # The viewer passes binary adjacency without self connections here.
    a = a + np.eye(len(a))
    d = a.sum(1)
    return a / np.sqrt(d[:, None] * d[None, :])
