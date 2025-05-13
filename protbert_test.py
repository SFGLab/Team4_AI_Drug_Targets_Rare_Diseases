# Example script to use ProtBert for protein sequence embedding and masked language modeling

from transformers import BertModel, BertTokenizer, BertForMaskedLM, BertTokenizer, pipeline
import re
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model_1 = BertModel.from_pretrained("Rostlab/prot_bert")
sequence_Example = "A E T C Z A O"
sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
encoded_input = tokenizer(sequence_Example, return_tensors='pt')
output = model_1    (**encoded_input)
print(output)
# Extract embeddings
embeddings = output.last_hidden_state  # shape: [batch_size, sequence_length, hidden_size]

# Print shape and size
print("Embedding tensor shape:", embeddings.shape)
print("Number of tokens (sequence length):", embeddings.shape[1])
print("Embedding dimension (hidden size):", embeddings.shape[2])

# Example#
model_2 = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
unmasker = pipeline('fill-mask', model=model_2, tokenizer=tokenizer)
print(unmasker('D L I P T S S K L V V [MASK] D T S L Q V K K A F F A L V T'))
