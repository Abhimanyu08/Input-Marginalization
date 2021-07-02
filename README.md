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


## Paper Summary:

Most of the techniques for calculating attribution scores for inputs to a model, zero out the input tokens entirely and then note the change in the predicted labels confidence. This method has one shortcoming: Replacing an input token id with 0 token is essentially replacing the tokens with `[PAD]` token. This input is completely out of distribution inputs to the model. There will never occur a sentence in the wild which has `[PAD]` token. This paper proposes to instead iteratively replace the token with each token from the vocabulary other than itself and note the change in confidence of predicted labels. We should do this with each token of sentence to calculate the effect of all the tokens on predicted label.
