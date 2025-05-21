import modal
import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

# Modal image with dependencies required for the original BaselineModel
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "rdkit-pypi",
        "pandas",
        "numpy",
        "scikit-learn"
    )
    # Note: torch_geometric is not needed for this BaselineModel architecture
)

# Define Modal volumes for input and output
# Assuming input data and the trained model are in my-hackathon-data and my-hackathon-outputs respectively
input_volume = modal.Volume.from_name("my-hackathon-data", create_if_missing=False)
output_volume = modal.Volume.from_name("my-hackathon-outputs", create_if_missing=True)

app = modal.App("ligand-protein-predict")

# BaselineModel definition (copied from training script)
# This definition should match the model you trained
class BaselineModel(nn.Module):
    def __init__(self, mol_feature_dim=15, protein_embed_dim=1024, hidden_dim=512, dropout=0.5):
        super(BaselineModel, self).__init__()
        self.mol_processor = nn.Sequential(
            nn.Linear(mol_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Updated to use precomputed embeddings
        self.protein_processor = nn.Sequential(
            nn.Linear(protein_embed_dim, 256),
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
        protein_repr = self.protein_processor(protein_features)
        combined = torch.cat([mol_repr, protein_repr], dim=1)
        output = self.fusion(combined)
        return output.squeeze()

def get_ligand_features(smiles, ligand_embeddings):
    """Get precomputed ligand features from embeddings dictionary."""
    if smiles in ligand_embeddings:
        return torch.tensor(ligand_embeddings[smiles], dtype=torch.float32)
    else:
        print(f"Ligand SMILES not found in embeddings: {smiles[:30]}...")
        return None

def get_protein_features(protein_seq, protein_embeddings):
    """Get precomputed protein features from embeddings dictionary."""
    if protein_seq in protein_embeddings:
        return torch.tensor(protein_embeddings[protein_seq], dtype=torch.float32)
    else:
        print(f"Protein sequence not found in embeddings: {protein_seq[:30]}...")
        return None

@app.function(gpu="A10G", timeout=600, image=image, volumes={"/data": input_volume, "/outputs": output_volume})
def predict_on_modal(
    input_csv_path,
    model_path,
    output_csv_path,
    protein_embeddings_path,
    ligand_embeddings_path
):
    print("=== Ligand-Protein Interaction Prediction (Modal) ===\n")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

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

    # Load embeddings
    try:
        protein_embeddings = np.load(protein_embeddings_path, allow_pickle=True).item()
        print(f"Loaded protein embeddings from: {protein_embeddings_path}")
        ligand_embeddings = np.load(ligand_embeddings_path, allow_pickle=True).item()
        print(f"Loaded ligand embeddings from: {ligand_embeddings_path}")
    except FileNotFoundError as e:
        print(f"Error: Embeddings file not found: {e}")
        return
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return

    assert 'SMILES' in df.columns and 'Protein' in df.columns, "Input CSV must have 'SMILES' and 'Protein' columns."
    has_labels = 'binding' in df.columns

    # Preprocess data
    smiles_features = []
    protein_features = []
    valid_indices = []
    for i, (smiles, protein) in enumerate(zip(df['SMILES'], df['Protein'])):
        mol_feat = get_ligand_features(smiles, ligand_embeddings)
        prot_feat = get_protein_features(protein, protein_embeddings)
        if mol_feat is not None and prot_feat is not None:
            smiles_features.append(mol_feat)
            protein_features.append(prot_feat)
            valid_indices.append(i)
    
    if not valid_indices:
        print("No valid samples found after preprocessing.")
        if output_csv_path:
             empty_df = pd.DataFrame(columns=['SMILES', 'Protein', 'predicted_prob', 'predicted_label'])
             empty_df.to_csv(output_csv_path, index=False)
             print(f"Saved empty predictions to {output_csv_path}")
        return

    # Prepare data for model
    X_mol = torch.stack(smiles_features).to(device)
    X_prot = torch.stack(protein_features).to(device)

    # Load model
    try:
        model = BaselineModel(
            mol_feature_dim=15,
            protein_embed_dim=1024,  # ProtBert embedding size
            hidden_dim=512,
            dropout=0.5
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Successfully loaded model from: {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    # Predict
    print("Making predictions...")
    with torch.no_grad():
        y_pred_prob = model(X_mol, X_prot).cpu().numpy()
        y_pred = (y_pred_prob > 0.5).astype(int)

    # Output predictions
    df_pred = df.iloc[valid_indices].copy()
    df_pred['predicted_prob'] = y_pred_prob
    df_pred['predicted_label'] = y_pred

    if output_csv_path:
        try:
            output_dir = os.path.dirname(output_csv_path)
            if output_dir and not os.path.exists(output_dir):
                 os.makedirs(output_dir, exist_ok=True)
            df_pred.to_csv(output_csv_path, index=False)
            print(f"Predictions saved to {output_csv_path}")
        except Exception as e:
             print(f"Error saving predictions to {output_csv_path}: {e}")
    else:
        print("Predictions (first 5 rows):")
        print(df_pred[['SMILES', 'Protein', 'predicted_prob', 'predicted_label']].head())

    # If labels are present, compute and print metrics
    if has_labels:
        y_true = df.iloc[valid_indices]['binding'].values
        try:
            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred_prob)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            
            # Create metrics dictionary
            metrics = {
                'metric': ['Accuracy', 'AUC-ROC', 'Precision', 'Recall', 'F1-Score'],
                'value': [acc, auc, precision, recall, f1]
            }
            
            # Create metrics DataFrame
            metrics_df = pd.DataFrame(metrics)
            
            # Save metrics to CSV
            metrics_path = os.path.join(os.path.dirname(output_csv_path), 'prediction_metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            print(f"\nPerformance metrics saved to: {metrics_path}")
            
            # Print metrics
            print("\nPerformance Metrics:")
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  AUC-ROC:   {auc:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            
            # Additional statistics
            stats = {
                'statistic': [
                    'Total Samples',
                    'Valid Samples',
                    'Positive Predictions',
                    'Negative Predictions',
                    'True Positives',
                    'True Negatives',
                    'False Positives',
                    'False Negatives'
                ],
                'value': [
                    len(df),
                    len(valid_indices),
                    sum(y_pred == 1),
                    sum(y_pred == 0),
                    sum((y_pred == 1) & (y_true == 1)),
                    sum((y_pred == 0) & (y_true == 0)),
                    sum((y_pred == 1) & (y_true == 0)),
                    sum((y_pred == 0) & (y_true == 1))
                ]
            }
            
            # Save statistics to CSV
            stats_df = pd.DataFrame(stats)
            stats_path = os.path.join(os.path.dirname(output_csv_path), 'prediction_statistics.csv')
            stats_df.to_csv(stats_path, index=False)
            print(f"Prediction statistics saved to: {stats_path}")
            
        except Exception as e:
            print(f"Error computing metrics: {e}")
    else:
        print("\nNo ground truth labels found. Performance metrics cannot be computed.")

# No local execution block needed for a Modal-focused prediction script
# Use `modal run predict_interaction_NN_modal.py predict_on_modal --help` to see arguments
# Use `modal run predict_interaction_NN_modal.py predict_on_modal --input-csv-path /data/your_input.csv --model-path /outputs/your_model.pth --output-csv-path /outputs/your_predictions.csv` to run 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict ligand-protein interactions using precomputed embeddings')
    parser.add_argument('--input-csv-path', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--output-csv-path', type=str, required=True,
                        help='Path to save predictions')
    parser.add_argument('--protein-embeddings-path', type=str, required=True,
                        help='Path to precomputed protein embeddings')
    parser.add_argument('--ligand-embeddings-path', type=str, required=True,
                        help='Path to precomputed ligand embeddings')
    args = parser.parse_args()
    
    predict_on_modal.remote(
        input_csv_path=args.input_csv_path,
        model_path=args.model_path,
        output_csv_path=args.output_csv_path,
        protein_embeddings_path=args.protein_embeddings_path,
        ligand_embeddings_path=args.ligand_embeddings_path
    ) 