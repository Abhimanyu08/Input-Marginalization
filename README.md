# Input-Marginalization
Unofficial PyTorch implementation of the paper "Interpretation of NLP models through input marginalization"


You can try it out in command line as follows:

```
python calculate_score.py\ 
--sent "I hated the movie"\
--label 0 
```
The following optional arguments can also be added:

* `--target_model`: Model you want to interpet. Default is `BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")`
* `--language_model`: Language model to be used in input marginalization. Default is `BertForMaskedLM.from_pretrained("bert-base-uncased")`
* `--tokenizer`: Tokenizer to use to preprocess the input sentence. Default is `BertTokenizer.from_pretrained("bert-base-uncased)`

