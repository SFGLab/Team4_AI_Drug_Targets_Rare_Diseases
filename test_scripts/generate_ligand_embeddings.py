import modal
import os
import argparse
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm

# Modal image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "rdkit-pypi",
        "pandas",
        "numpy",
        "tqdm"
    )
)

# Define Modal volumes
input_volume = modal.Volume.from_name("my-hackathon-data", create_if_missing=False)
output_volume = modal.Volume.from_name("my-hackathon-outputs", create_if_missing=True)

app = modal.App("ligand-embeddings-generator")

def smiles_to_features(smiles):
    """Convert SMILES string to molecular features using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES skipped: {smiles}")
            return None
        features = [
            rdMolDescriptors.CalcExactMolWt(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            mol.GetNumAtoms(),
            mol.GetNumBonds(),
            Chem.rdMolDescriptors.CalcNumAliphaticCarbocycles(mol),
            Chem.rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
            Chem.rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),
            Chem.rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
            rdMolDescriptors.CalcNumHeteroatoms(mol),
            rdMolDescriptors.CalcNumSaturatedRings(mol)
        ]
        return np.array(features, dtype=np.float32)
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None

@app.function(gpu="A10G", timeout=600, image=image, volumes={"/data": input_volume, "/outputs": output_volume})
def generate_ligand_embeddings(
    input_csv_path="/data/ligands.csv",
    output_path="/outputs/ligand_embeddings.npy",
    smiles_column="SMILES"
):
    """
    Generate ligand embeddings from SMILES strings in a CSV file.
    
    Args:
        input_csv_path (str): Path to input CSV file containing SMILES strings
        output_path (str): Path to save the embeddings
        smiles_column (str): Name of the column containing SMILES strings
    """
    print("=== Ligand Embeddings Generation ===\n")

    # Load data
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Loaded input data from: {input_csv_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        return
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    # Verify SMILES column exists
    if smiles_column not in df.columns:
        print(f"Error: Column '{smiles_column}' not found in CSV file")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings_dict = {}
    skipped_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row[smiles_column]
        features = smiles_to_features(smiles)
        if features is not None:
            embeddings_dict[smiles] = features
        else:
            skipped_count += 1

    # Save embeddings
    try:
        np.save(output_path, embeddings_dict)
        print(f"\nEmbeddings saved to: {output_path}")
        print(f"Successfully processed {len(embeddings_dict)} SMILES strings")
        print(f"Skipped {skipped_count} invalid SMILES strings")
        
        # Print some statistics about the embeddings
        if embeddings_dict:
            first_embedding = next(iter(embeddings_dict.values()))
            print(f"\nEmbedding dimension: {len(first_embedding)}")
            print("Feature names:")
            print("1. Molecular Weight")
            print("2. TPSA (Topological Polar Surface Area)")
            print("3. Number of H-Bond Donors")
            print("4. Number of H-Bond Acceptors")
            print("5. Number of Rotatable Bonds")
            print("6. Number of Aromatic Rings")
            print("7. Fraction of SP3 Carbon Atoms")
            print("8. Number of Atoms")
            print("9. Number of Bonds")
            print("10. Number of Aliphatic Carbocycles")
            print("11. Number of Aliphatic Heterocycles")
            print("12. Number of Saturated Carbocycles")
            print("13. Number of Saturated Heterocycles")
            print("14. Number of Heteroatoms")
            print("15. Number of Saturated Rings")
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ligand embeddings from SMILES strings')
    parser.add_argument('--input-csv-path', type=str, default="/data/ligands.csv",
                        help='Path to input CSV file containing SMILES strings')
    parser.add_argument('--output-path', type=str, default="/outputs/ligand_embeddings.npy",
                        help='Path to save the embeddings')
    parser.add_argument('--smiles-column', type=str, default="SMILES",
                        help='Name of the column containing SMILES strings')
    args = parser.parse_args()
    
    generate_ligand_embeddings.remote(
        input_csv_path=args.input_csv_path,
        output_path=args.output_path,
        smiles_column=args.smiles_column
    ) 