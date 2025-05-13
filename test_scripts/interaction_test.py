# Simple script to test interaction between protein and drug. 

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
import re

# ========== 1. Load Models and Tokenizers ==========

# ProtBERT
prot_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
prot_model = BertModel.from_pretrained("Rostlab/prot_bert")

# ChemBERTa
chem_tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
chem_model = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

prot_model.eval()
chem_model.eval()

# ========== 2. Define Simple MLP Model ==========

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
        x = torch.sigmoid(self.out(x))
        return x

# ========== 3. Embedding Utility Functions ==========

def embed_protein(sequence: str):
    sequence = re.sub(r"[UZOB]", "X", sequence)
    tokens = prot_tokenizer(sequence, return_tensors='pt')
    with torch.no_grad():
        output = prot_model(**tokens)
    return output.last_hidden_state.mean(dim=1)  # shape: [1, 1024]

def embed_smiles(smiles: str):
    tokens = chem_tokenizer(smiles, return_tensors='pt')
    with torch.no_grad():
        output = chem_model(**tokens)
    return output.last_hidden_state.mean(dim=1)  # shape: [1, 768]

# ========== 4. Example Input ==========

protein_seq = "A E T C Z A O"
smiles_str = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin

prot_embedding = embed_protein(protein_seq)  # [1, 1024]
chem_embedding = embed_smiles(smiles_str)    # [1, 768]

# ========== 5. Run through Model ==========

model = ProteinLigandNN()
output = model(prot_embedding, chem_embedding)

print("Predicted interaction score (0–1):", output.item())
