import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

# BaselineModel definition (copied from training script)
class BaselineModel(nn.Module):
    def __init__(self, mol_feature_dim=15, protein_vocab_size=22, protein_embed_dim=128, 
                 protein_max_len=1000, hidden_dim=512, dropout=0.5):
        super(BaselineModel, self).__init__()
        self.mol_processor = nn.Sequential(
            nn.Linear(mol_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.protein_embedding = nn.Embedding(protein_vocab_size, protein_embed_dim, padding_idx=0)
        self.protein_lstm = nn.LSTM(protein_embed_dim, 256, batch_first=True, bidirectional=True)
        self.protein_processor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fusion = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, mol_features, protein_features):
        mol_repr = self.mol_processor(mol_features)
        protein_embedded = self.protein_embedding(protein_features)
        protein_lstm_out, _ = self.protein_lstm(protein_embedded)
        protein_repr = torch.mean(protein_lstm_out, dim=1)
        protein_repr = self.protein_processor(protein_repr)
        combined = torch.cat([mol_repr, protein_repr], dim=1)
        output = self.fusion(combined)
        return output.squeeze()

def smiles_to_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
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
        return torch.tensor(features, dtype=torch.float32)
    except:
        return None

def protein_to_features(protein_seq, max_protein_len=1000):
    aa_vocab = {
        'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'E': 6, 'Q': 7, 'G': 8,
        'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
        'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20, 'X': 21
    }
    indices = [aa_vocab.get(aa, aa_vocab['X']) for aa in protein_seq]
    if len(indices) > max_protein_len:
        indices = indices[:max_protein_len]
    else:
        indices.extend([0] * (max_protein_len - len(indices)))
    return torch.tensor(indices, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser(description='Predict ligand-protein interactions using a trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model .pth file')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file (with SMILES, Protein, and optionally binding columns)')
    parser.add_argument('--output', type=str, help='Path to save predictions as CSV')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    assert 'SMILES' in df.columns and 'Protein' in df.columns, "Input CSV must have 'SMILES' and 'Protein' columns."
    has_labels = 'binding' in df.columns

    # Preprocess
    smiles_features = []
    protein_features = []
    valid_indices = []
    for i, (smiles, protein) in enumerate(zip(df['SMILES'], df['Protein'])):
        mol_feat = smiles_to_features(smiles)
        prot_feat = protein_to_features(protein)
        if mol_feat is not None and prot_feat is not None:
            smiles_features.append(mol_feat)
            protein_features.append(prot_feat)
            valid_indices.append(i)
    if not valid_indices:
        print("No valid samples found in input.")
        return
    X_mol = torch.stack(smiles_features)
    X_prot = torch.stack(protein_features)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaselineModel()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    # Predict
    with torch.no_grad():
        y_pred_prob = model(X_mol.to(device), X_prot.to(device)).cpu().numpy()
        y_pred = (y_pred_prob > 0.5).astype(int)

    # Output predictions
    df_pred = df.iloc[valid_indices].copy()
    df_pred['predicted_prob'] = y_pred_prob
    df_pred['predicted_label'] = y_pred
    if args.output:
        df_pred.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
    else:
        print(df_pred[['SMILES', 'Protein', 'predicted_prob', 'predicted_label']].head())

    # If labels are present, compute metrics
    if has_labels:
        y_true = df.iloc[valid_indices]['binding'].values
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_prob)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        print("Performance Metrics:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    else:
        print("No ground truth labels found. Only predictions are shown.")

if __name__ == "__main__":
    main() 