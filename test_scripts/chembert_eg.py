# Test script to use ChemBERT for SMILES sequence embedding and feed into a NN model
from transformers import RobertaTokenizer, RobertaModel
import torch

# Load ChemBERTa tokenizer and model (pretrained on SMILES)
tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# Example SMILES string
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin

# Tokenize the SMILES
encoded_input = tokenizer(smiles, return_tensors='pt')

# Disable gradients for inference
with torch.no_grad():
    output = model(**encoded_input)

# Output embeddings: last_hidden_state shape: [batch_size, sequence_length, hidden_size]
embeddings = output.last_hidden_state

# Option 1: CLS token embedding (recommended for classification tasks)
cls_embedding = embeddings[:, 0, :]  # shape: [1, hidden_size]
print("CLS Embedding shape:", cls_embedding.shape)

# Option 2: Mean pooling (useful for general representation)
mean_embedding = embeddings.mean(dim=1)  # shape: [1, hidden_size]
print("Mean Embedding shape:", mean_embedding.shape)
