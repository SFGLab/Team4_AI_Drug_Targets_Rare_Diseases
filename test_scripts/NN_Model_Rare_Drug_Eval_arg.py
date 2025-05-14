#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 18:32:11 2025

@author: dliu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import Accuracy
from torchmetrics.functional import auroc
import pandas as pd
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import argparse
import re

# ========== 1. Load Pretrained Models, Tokenizers and Parameters ==========
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="path to model")
parser.add_argument("--data", type=str, help="dataset path")
parser.add_argument("--batch", type=int, help="size of training batches")
args = parser.parse_args()

model_path = "RARExDrug_NN_7000recs.pth"

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
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path).dropna(subset=["SMILES", "Protein", "binding"])
        self.samples = df[["SMILES", "Protein", "binding"]].reset_index(drop=True)

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

# evaluate model
def evaluate(path):
    model = torch.load(path)
    
    model.eval()
    
    dataset = ProteinLigandDataset(args.data)
#    dataset = ProteinLigandDataset("/Users/dliu/Desktop/test_folders/workdir/Team4/BindingDB_ChEMBL_LiganSmile_ProteinSeq_750k_Binding_noBinding_3k_Val.csv")
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False)    

    with torch.no_grad():
        for prot_embed,  chem_embed, labels in loader:

            # make predictions
            outputs = model(prot_embed,chem_embed)
            print(outputs)
            
            # loss function
            loss_fn = nn.BCELoss()
            loss = loss_fn(outputs, labels)
            print("validation binary cross entropy loss:",loss)
            
            accuracy = Accuracy(task="binary", num_classes=2)
            acc = accuracy(outputs, labels)
            print("validation accuracy:",acc)
            
            # create confusion matrix
            confmat = ConfusionMatrix(task="binary",num_classes=2)
            matrix = confmat(outputs, labels)
            print("confusion matrix:\n[TN,FP]\n[FN,TP]\n", matrix)
            
            # create AUOROC
            auc_score = auroc(outputs, target=labels.int(), task="binary",num_classes=2)
            print("auroc score:",auc_score)

            fpr, tpr, thresholds = roc_curve(labels, outputs)
            # print("False Positive Rate:", fpr)
            # print("True Positive Rate:", tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc='lower right')
            plt.show()
            
if __name__ == "__main__":
    
    evaluate(model_path)
