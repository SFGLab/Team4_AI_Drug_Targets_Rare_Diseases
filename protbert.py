# Example script to use ProtBert for protein sequence embedding and masked language modeling

from transformers import BertModel, BertTokenizer, BertForMaskedLM, BertTokenizer, pipeline
import re
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model = BertModel.from_pretrained("Rostlab/prot_bert")
sequence_Example = "A E T C Z A O"
sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
encoded_input = tokenizer(sequence_Example, return_tensors='pt')
output = model(**encoded_input)
print(output)

# Example#
model2 = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
unmasker = pipeline('fill-mask', model=model2, tokenizer=tokenizer)
print(unmasker('D L I P T S S K L V V [MASK] D T S L Q V K K A F F A L V T'))
