import modal
import pandas as pd
import torch
import numpy as np
from transformers import BertModel, BertTokenizer

# Modal image with transformers and sentencepiece
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "sentencepiece",
        "pandas",
        "numpy"
    )
)

# Modal volume for input/output
volume = modal.Volume.from_name("my-hackathon-data", create_if_missing=False)

app = modal.App("precompute-protein-embeddings")

@app.function(image=image, volumes={"/data": volume}, timeout=3600, gpu="A10G")
def precompute_embeddings_modal(csv_path="/data/BindingDB_ChEMBL_LiganSmile_ProteinSeq_750k_Binding_noBinding_7k_Train.csv", output_path="/data/protein_embeddings.npy", batch_size=8):
    print(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    unique_proteins = df['Protein'].unique()
    print(f"Found {len(unique_proteins)} unique protein sequences.")

    print("Loading ProtBert model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    model.eval()

    embeddings = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)

    # Batch processing
    for i in range(0, len(unique_proteins), batch_size):
        batch_seqs = unique_proteins[i:i+batch_size]
        batch_spaced = [' '.join(list(seq)) for seq in batch_seqs]
        encoded_input = tokenizer(
            batch_spaced,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            output = model(**encoded_input)
        # CLS token for each sequence
        batch_embs = output.last_hidden_state[:, 0, :].cpu().numpy()  # (batch, 1024)
        for seq, emb in zip(batch_seqs, batch_embs):
            embeddings[seq] = emb
        print(f"Processed {i+len(batch_seqs)}/{len(unique_proteins)} proteins")

    np.save(output_path, embeddings)
    print(f"Saved protein embeddings to {output_path}")

if __name__ == "__main__":
    # For local test, you could call precompute_embeddings_modal.remote() with args
    pass 