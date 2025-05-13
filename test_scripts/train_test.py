# Test script to train a simple MLP model on a dataset of protein-ligand interactions. import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader
import re
from tqdm import tqdm

# ========== 1. Load Pretrained Models and Tokenizers ==========

prot_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
prot_model = BertModel.from_pretrained("Rostlab/prot_bert")
prot_model.eval()

chem_tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
chem_model = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
chem_model.eval()

# ========== 2. MLP Class ==========

class ProteinLigandNN(nn.Module):
    def __init__(self):
        super(ProteinLigandNN, self).__init__()
        self.fc1 = nn.Linear(1024 + 768, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, prot_embed, chem_embed):
        x = torch.cat([prot_embed, chem_embed], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.out(x))

# ========== 3. Data Handling ==========

class ProteinLigandDataset(Dataset):
    def __init__(self, csv_path, max_samples=100):
        df = pd.read_csv(csv_path).dropna(subset=["SMILES", "Protein", "binding"])
        self.samples = df.head(max_samples)[["SMILES", "Protein", "binding"]].reset_index(drop=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        smiles = self.samples.loc[idx, "SMILES"]
        prot_seq = re.sub(r"[UZOB]", "X", self.samples.loc[idx, "Protein"])
        label = float(self.samples.loc[idx, "binding"])
        
        # Chem embedding
        chem_tokens = chem_tokenizer(smiles, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            chem_embed = chem_model(**chem_tokens).last_hidden_state.mean(dim=1).squeeze(0)

        # Protein embedding
        prot_tokens = prot_tokenizer(prot_seq, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            prot_embed = prot_model(**prot_tokens).last_hidden_state.mean(dim=1).squeeze(0)

        return prot_embed, chem_embed, torch.tensor([label], dtype=torch.float32)

# ========== 4. Training Loop ==========

def train():
    dataset = ProteinLigandDataset("/Users/mukulsherekar/pythonProject/Team4/input_files/BindingDB_ChEMBL_LiganSmile_ProteinSeq_750k_Binding_noBinding.csv")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = ProteinLigandNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()

    model.train()
    for epoch in range(5):
        total_loss = 0
        for prot_embed, chem_embed, labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
            preds = model(prot_embed, chem_embed)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

if __name__ == "__main__":
    train()