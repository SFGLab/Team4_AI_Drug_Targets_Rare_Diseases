# Team4_AI_Drug_Targets_Rare_Diseases
Main project tasks:
1. Find a relevant rare-disease for which genetic information are available regarding the existing variants

We will focus on cystic fibrosis that is common among the world populations but still considered rare.
Drug treatment of cystic fibrosis mainly targets the CFTR gene (Cystic Fibrosis Transmembrane Conductance Regulator). Drugs aim to correct the expression expression or function of the defective CFTR protein. The relevance of CFTR for th treatment of cystic fibrosis is also remarked by the number of pathogenic or likely-pathogenic variants hosted on Clinvar (https://www.ncbi.nlm.nih.gov/clinvar) that are linked to this gene. Based on Clinvar, CFTR presents 1102 Pathogenic/Likely-pathogenic variants. These 1102 variants include the following main variant types:
| **Mutation type**    | **Count**                  |
| ----------------- | --------------------------- |
| Deletion | 404 |
| Duplication | 127 |
| Indel | 28 |
| Insertion | 10 |
| Microsatellite | 20 |
| Copy number loss | 1 |
| single nucleotide variant | 504 |


Drugs currently available to treat cystic fibrosis are the following:

| **Drug Class**    | **Target**                  | **Example**                     |
| ----------------- | --------------------------- | ------------------------------- |
| Potentiator       | CFTR gating                 | Ivacaftor                       |
| Corrector         | CFTR folding/trafficking    | Lumacaftor, Tezacaftor          |
| Amplifier         | CFTR expression             | Nesolicaftor (experimental)     |
| Readthrough agent | Premature stop codons       | Ataluren                        |
| Gene therapy      | CFTR gene                   | CRISPR, mRNA delivery           |
| ENaC inhibitors   | Sodium channel              | BI 1265162 (investigational)    |
| Anti-inflammatory | Neutrophilic inflammation   | Lenabasum, ibuprofen            |
| Mucolytics        | Mucus degradation/hydration | Dornase alfa, hypertonic saline |

We obtained the chemical structure of Ivacaftor and we plan to obtain the same information for others drugs that targets cystic fibrosis with the aim of comparing the structure of the drugs currently used to treat cystic fibrosis with the ligand that will be identified through our machine learning model.

3. Select a database of protein-ligand interactions to be used as input for the training of the ML model
4. Find a tool to generate embeddings from the protein sequences
5. Build a first-prototype of the deep learning model using a basic architecture
