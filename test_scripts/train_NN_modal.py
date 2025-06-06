import modal
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define Modal image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib"
    )
)

# Define the Modal volume for input
volume = modal.Volume.from_name("my-hackathon-data", create_if_missing=False)
# Define the Modal volume for output
output_volume = modal.Volume.from_name("my-hackathon-outputs", create_if_missing=True)

class LigandProteinDataset(Dataset):
    """Dataset class for ligand-protein interaction data using precomputed embeddings"""
    def __init__(self, smiles_list, protein_list, labels, protein_embeddings_path, ligand_embeddings_path):
        self.smiles_list = smiles_list
        self.protein_list = protein_list
        self.labels = labels
        
        # Load embeddings
        self.protein_embeddings = np.load(protein_embeddings_path, allow_pickle=True).item()
        self.ligand_embeddings = np.load(ligand_embeddings_path, allow_pickle=True).item()
        
        self.mol_features = []
        self.protein_features = []
        self.valid_indices = []
        
        for i, (smiles, protein, label) in enumerate(zip(smiles_list, protein_list, labels)):
            mol_feat = self._get_ligand_features(smiles)
            prot_feat = self._get_protein_features(protein)
            if mol_feat is not None and prot_feat is not None:
                self.mol_features.append(mol_feat)
                self.protein_features.append(prot_feat)
                self.valid_indices.append(i)
        
        self.labels = [labels[i] for i in self.valid_indices]
        print(f"Processed {len(self.valid_indices)} valid samples out of {len(smiles_list)}")
    
    def _get_ligand_features(self, smiles):
        """Get precomputed ligand features from embeddings dictionary."""
        if smiles in self.ligand_embeddings:
            return torch.tensor(self.ligand_embeddings[smiles], dtype=torch.float32)
        else:
            print(f"Ligand SMILES not found in embeddings: {smiles[:30]}...")
            return None
    
    def _get_protein_features(self, protein_seq):
        """Get precomputed protein features from embeddings dictionary."""
        if protein_seq in self.protein_embeddings:
            return torch.tensor(self.protein_embeddings[protein_seq], dtype=torch.float32)
        else:
            print(f"Protein sequence not found in embeddings: {protein_seq[:30]}...")
            return None
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        return {
            'mol_features': self.mol_features[idx],
            'protein_features': self.protein_features[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

class BaselineModel(nn.Module):
    """Baseline model using simple neural networks"""
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

class ModelTrainer:
    """Trainer class for the models"""
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_aucs = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []
    
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        for batch in train_loader:
            optimizer.zero_grad()
            mol_features = batch['mol_features'].to(self.device)
            protein_features = batch['protein_features'].to(self.device)
            labels = batch['label'].to(self.device)
            outputs = self.model(mol_features, protein_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predictions.extend((outputs > 0.5).cpu().numpy())
            targets.extend(labels.cpu().numpy())
        accuracy = accuracy_score(targets, predictions)
        return total_loss / len(train_loader), accuracy
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        probs = []
        with torch.no_grad():
            for batch in val_loader:
                mol_features = batch['mol_features'].to(self.device)
                protein_features = batch['protein_features'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(mol_features, protein_features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                probs.extend(outputs.cpu().numpy())
                predictions.extend((outputs > 0.5).cpu().numpy())
                targets.extend(labels.cpu().numpy())
        accuracy = accuracy_score(targets, predictions)
        auc = roc_auc_score(targets, probs)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='binary')
        return total_loss / len(val_loader), accuracy, auc, precision, recall, f1
    
    def train(self, train_loader, val_loader, num_epochs=100, lr=0.001):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        best_val_auc = 0
        patience_counter = 0
        patience = 5
        epochs_completed = 0
        
        print("Starting training with fresh model")
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc, val_auc, val_precision, val_recall, val_f1 = self.validate(val_loader, criterion)
            scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.val_aucs.append(val_auc)
            self.val_precisions.append(val_precision)
            self.val_recalls.append(val_recall)
            self.val_f1s.append(val_f1)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), '/outputs/best_model.pth')
                print(f"Saved new best model with AUC: {best_val_auc:.4f}")
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}:')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')
                print(f'  Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
            
            epochs_completed += 1
        
        print(f"Total epochs completed: {epochs_completed}")
        return best_val_auc

def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Missing values per column:")
    print(df.isnull().sum())
    
    required_cols = ['SMILES', 'Protein', 'binding']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        df.columns = df.columns.str.strip()
        for col in required_cols:
            matches = [c for c in df.columns if c.lower() == col.lower()]
            if matches:
                df = df.rename(columns={matches[0]: col})
                print(f"Renamed {matches[0]} to {col}")
    
    print(f"Binding distribution: {df['binding'].value_counts()}")
    print(f"Sample of data:")
    print(df.head())
    print(f"\nData quality checks:")
    print(f"Empty SMILES: {(df['SMILES'].str.len() == 0).sum()}")
    print(f"Empty Proteins: {(df['Protein'].str.len() == 0).sum()}")
    print(f"Very short SMILES (<5 chars): {(df['SMILES'].str.len() < 5).sum()}")
    print(f"Very short Proteins (<10 chars): {(df['Protein'].str.len() < 10).sum()}")
    
    return df['SMILES'].tolist(), df['Protein'].tolist(), df['binding'].tolist()

# Modal App
app = modal.App("hackathon-train-nn")

@app.function(gpu="A10G", timeout=3600, image=image, volumes={"/data": volume, "/outputs": output_volume})
def modal_main(
    dataset_path,
    protein_embeddings_path,
    ligand_embeddings_path
):
    print("=== Ligand-Protein Interaction Prediction (Modal) ===\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    print(f"Loading dataset from: {dataset_path}")
    smiles_list, protein_list, labels = load_and_preprocess_data(dataset_path)
    
    print("Creating dataset...")
    dataset = LigandProteinDataset(
        smiles_list, 
        protein_list, 
        labels,
        protein_embeddings_path,
        ligand_embeddings_path
    )
    
    if len(dataset) == 0:
        print("Error: No valid samples in dataset. Check your data format.")
        return None, None
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    if train_size == 0:
        train_size = 1
        val_size = len(dataset) - 1
    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")
    
    print("=== Training Baseline Model ===")
    baseline_model = BaselineModel(
        mol_feature_dim=15,
        protein_embed_dim=1024,  # ProtBert embedding size
        hidden_dim=512,
        dropout=0.5
    )
    
    baseline_trainer = ModelTrainer(baseline_model, device)
    baseline_auc = baseline_trainer.train(train_loader, val_loader, num_epochs=50)
    print(f"Best baseline validation AUC: {baseline_auc:.4f}\n")

    # Save training metrics to CSV
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, len(baseline_trainer.train_losses) + 1)),
        'train_loss': baseline_trainer.train_losses,
        'train_acc': baseline_trainer.train_accuracies,
        'val_loss': baseline_trainer.val_losses,
        'val_acc': baseline_trainer.val_accuracies,
        'val_auc': baseline_trainer.val_aucs,
        'val_precision': baseline_trainer.val_precisions,
        'val_recall': baseline_trainer.val_recalls,
        'val_f1': baseline_trainer.val_f1s
    })
    
    output_metrics_path = "/outputs/training_metrics.csv"
    metrics_df.to_csv(output_metrics_path, index=False)
    print(f"Training metrics saved as {output_metrics_path}")

    output_path = "/outputs/final_baseline_model.pth"
    torch.save(baseline_model.state_dict(), output_path)
    print(f"Model saved as {output_path}")
    return baseline_auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ligand-protein interaction model using precomputed embeddings')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to training dataset CSV file')
    parser.add_argument('--protein-embeddings-path', type=str, required=True,
                        help='Path to precomputed protein embeddings')
    parser.add_argument('--ligand-embeddings-path', type=str, required=True,
                        help='Path to precomputed ligand embeddings')
    args = parser.parse_args()
    
    modal_main.remote(
        dataset_path=args.dataset_path,
        protein_embeddings_path=args.protein_embeddings_path,
        ligand_embeddings_path=args.ligand_embeddings_path
    ) 