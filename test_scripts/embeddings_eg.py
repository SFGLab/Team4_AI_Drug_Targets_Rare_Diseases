"""
Protein-Substrate Graph Embedding using GNNs 
=================================================
This script constructs molecular graphs for protein-ligand complexes and extracts 
embeddings using a Graph Neural Network (GNN).
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data
from rdkit import Chem
from Bio import PDB
import numpy as np

# Function to extract atomic features (One-Hot Encoding)
def atom_features(atom):
    atom_type = ['C', 'N', 'O', 'S', 'P', 'H', 'F', 'Cl', 'Br', 'I']
    return torch.tensor([int(atom.GetSymbol() == x) for x in atom_type], dtype=torch.float)

# Function to extract bond features (One-Hot Encoding)
def bond_features(bond):
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    return torch.tensor([int(bond.GetBondType() == x) for x in bond_types], dtype=torch.float)

# Function to construct a molecular graph from an RDKit molecule
def mol_to_graph(mol):
    atom_features_list = [atom_features(atom) for atom in mol.GetAtoms()]
    edge_index = []
    edge_features_list = []
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # Bidirectional edges
        edge_features_list.append(bond_features(bond))
        edge_features_list.append(bond_features(bond))
    
    x = torch.stack(atom_features_list, dim=0)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_features_list, dim=0)
    
    print(f"Molecular Graph Created: {mol.GetNumAtoms()} atoms, {len(edge_index[0])} edges")
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Function to convert a PDB protein structure into a graph
def pdb_to_graph(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    atom_features_list = []
    edge_index = []
    atom_list = []

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_list.append(atom)

    num_atoms = len(atom_list)
    atom_type = ['C', 'N', 'O', 'S', 'P', 'H', 'F', 'Cl', 'Br', 'I']

    for i in range(num_atoms):
        atom_i = atom_list[i]
        atom_symbol = atom_i.element if hasattr(atom_i, 'element') else 'C'
        atom_features_list.append(torch.tensor([int(atom_symbol == x) for x in atom_type], dtype=torch.float))

        for j in range(i + 1, num_atoms):
            atom_j = atom_list[j]
            distance = atom_i - atom_j
            if distance < 5.0:  
                edge_index.append([i, j])
                edge_index.append([j, i])
    
    x = torch.stack(atom_features_list, dim=0)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.ones((edge_index.shape[1], 4))  

    print(f"Protein Graph Created: {num_atoms} atoms, {len(edge_index[0])} edges")
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Define GNN Model for Protein-Ligand Embedding
class ProteinLigandGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ProteinLigandGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        batch = torch.zeros(x.size(0), dtype=torch.long)
        return global_mean_pool(self.conv2(x, edge_index), batch)

# Main Execution
if __name__ == "__main__":
    print("\n**Protein-Ligand Graph Embedding Pipeline** 🔬\n")
    
    # Input: Replace these with actual files and SMILES
    pdb_file = "6k4d.pdb"  # Protein file (PDB)
    ligand_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ligand (SMILES)
    
    print("Loading Protein Structure...")
    protein_graph = pdb_to_graph(pdb_file)

    print("Converting Ligand SMILES to Graph...")
    mol = Chem.MolFromSmiles(ligand_smiles)
    ligand_graph = mol_to_graph(mol)

    # Combine protein and ligand graphs into one
    print("🔗 Merging Protein and Ligand Graphs...")
    protein_ligand_graph = Data(
        x=torch.cat([protein_graph.x, ligand_graph.x], dim=0),
        edge_index=torch.cat([protein_graph.edge_index, ligand_graph.edge_index + len(protein_graph.x)], dim=1),
        edge_attr=torch.cat([protein_graph.edge_attr, ligand_graph.edge_attr], dim=0)
    )

    print(f"Combined Graph Created: {protein_ligand_graph.x.shape[0]} nodes, {protein_ligand_graph.edge_index.shape[1]} edges")

    # 🏗️ Initialize and Run GNN Model
    print("\nRunning GNN for Embedding Extraction...\n")
    gnn = ProteinLigandGNN(in_channels=10, hidden_channels=64, out_channels=32)
    embedding = gnn(protein_ligand_graph)

    # Save the final embedding
    torch.save(embedding, "protein_ligand_embedding.pt")
    print("**Embedding Saved Successfully!**\n")

    # Display the final embedding
    print(f"**Protein-Ligand Embedding Shape:** {embedding.shape}\n")
    print("**Pipeline Completed Successfully!** ")
