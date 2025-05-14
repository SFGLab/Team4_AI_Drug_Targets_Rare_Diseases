configfile: "config.yaml"

rule all:
    input:
        "models/trained_model.h5"

rule generate_embeddings:
    input:
        csv="data/input.csv"
    output:
        protein="data/protein_embeddings.npy",
        ligand="data/ligand_embeddings.npy"
    container:
        config["container"]
    shell:
        "python scripts/embeddings_eg.py --input {input.csv} --protein_out {output.protein} --ligand_out {output.ligand}"

rule train_model:
    input:
        protein="data/protein_embeddings.npy",
        ligand="data/ligand_embeddings.npy"
    output:
        model="models/trained_model.h5",
        preds="models/predictions.csv"
    container:
        config["container"]
    shell:
        "python scripts/NN_Model_Rare_Drug_Pred.py \
        --protein_embeddings {input.protein} \
        --ligand_embeddings {input.ligand} \
        --output_model {output.model} \
        --output_preds {output.preds}"

