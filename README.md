# Input-Marginalization
Unofficial PyTorch implementation of the paper "Interpretation of NLP models through input marginalization"


You can try it out in command line as follows:

```
python calculate_score.py\ 
--sent "I hated the movie"\
--label 0 
--target_model BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2") \
--language_model BertForMaskedLM.from_pretrained("bert-base-uncased")\
--tokenizer BertTokenizer.from_pretrained("bert-base-uncased)
```
