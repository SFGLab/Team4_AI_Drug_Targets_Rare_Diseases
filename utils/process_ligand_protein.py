import pandas as pd
import random
import argparse

# Define the mutation function
def mutate_protein(seq, mutation_rate=0.3):
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    seq = list(seq)
    n_mutate = int(len(seq) * mutation_rate)
    
    indices = random.sample(range(len(seq)), n_mutate)
    
    for i in indices:
        original = seq[i]
        choices = [aa for aa in amino_acids if aa != original]
        seq[i] = random.choice(choices)
    
    return ''.join(seq)

# Main function to process input
def main(input_file, output_file):
    # Read the input file
    df = pd.read_csv(input_file, sep='\t', header=None, names=['SMILES', 'Protein'])

    # Subsample 250,000 rows
    df_subsample = df.sample(n=250000, random_state=42).reset_index(drop=True)

    # Mutate protein sequences
    df_subsample['Mutated_Protein'] = df_subsample['Protein'].apply(mutate_protein)

    # Add "binding" column with all zero values
    df_subsample['binding'] = 0

    # Save to output file
    df_subsample.to_csv(output_file, index=False)

    print(f"Processed data saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ligand-protein pairs.")
    parser.add_argument("input_file", help="Path to input file (tab-separated, no header)")
    parser.add_argument("output_file", help="Path to save processed output CSV")
    args = parser.parse_args()
    
    main(args.input_file, args.output_file)

