
# Script to use ProtBert for protein sequence embedding and feed into a NN model
# This script provides a basic example of how to use ProtBert for protein sequence embedding.
# It demonstrates how to tokenize a protein sequence, pass it through the ProtBert model,
# and extract residue-wise embeddings.
# For a simple classification task. 

from transformers import BertModel, BertTokenizer
import torch
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")

# Example protein sequence
sequence = "A E T C Z A O"
sequence = re.sub(r"[UZOB]", "X", sequence)

# Tokenize and pass through model
encoded_input = tokenizer(sequence, return_tensors='pt')
with torch.no_grad():
    output = model(**encoded_input)

# Get residue-wise embeddings: shape (1, seq_len, 1024)
residue_embeddings = output.last_hidden_state

# Take mean across token dimension (dim=1): shape becomes (1, 1024)
# Mean embeddings becasue protein length is variable. 
mean_embedding = residue_embeddings.mean(dim=1)

# Convert to NumPy for Keras input
embedding_np = mean_embedding.cpu().numpy()  # shape: (1, 1024)

class ProteinMLP(nn.Module):
    def __init__(self, input_dim=1024):
        super(ProteinMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))    # Layer 1
        x = F.relu(self.fc2(x))    # Layer 2
        x = F.relu(self.fc3(x))    # Layer 3
        x = torch.sigmoid(self.out(x))  # Output for binary classification
        return x

# Initialize model
model = ProteinMLP(input_dim=1024)

# Dummy input (batch of 10 mean embeddings from ProtBERT)
x = torch.randn(10, 1024)  # shape: [batch_size, embedding_dim]

# Forward pass
output = model(x)
print(output.shape) 
print(output)  
