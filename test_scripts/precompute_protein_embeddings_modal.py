import modal
import pandas as pd
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
import argparse

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
def precompute_embeddings_modal(input_dataset: str, output_path="/data/protein_embeddings.npy", batch_size=8):
    print(f"Loading CSV from {input_dataset}")
    df = pd.read_csv(input_dataset)
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
    parser = argparse.ArgumentParser(description="Precompute protein embeddings with ProtBert")
    parser.add_argument('--input_dataset', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_path', type=str, default="protein_embeddings.npy", help='Path to output .npy file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for embedding generation')
    args = parser.parse_args()
    print(f"Running locally with input CSV: {args.input_dataset}")
    precompute_embeddings_modal(args.input_dataset, args.output_path, args.batch_size)