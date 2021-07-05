import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from typing import List,Union, Dict
import torch.nn.functional as F
import numpy as np
import transformers
# from numba import prange, jit
from torch import Tensor

class CustomDataset():
	def __init__(self, input_ids, attention_masks, token_type_ids):
		self.input_ids = input_ids
		self.attention_masks = attention_masks
		self.token_type_ids = token_type_ids

	def __len__(self): return self.input_ids.size(0)
	

	def __getitem__(self, i): return {'input_ids': self.input_ids[i,:], 'attention_mask': self.attention_masks[i,:], 'token_type_ids': self.token_type_ids[i, :]}


def calculate_masked_probabilities(masked_lm: transformers.PreTrainedModel, sample: Dict, index: int) -> Tensor:
	'''
	Arguments:
		masked_lm -> A language model good at MAKSED language modelling for eg. BERT
		sample -> A Dictionary containing 'input_ids', 'attention_mask', 'token_type_ids' as keys. 
		index -> Index of the masked token in sequence. for eg if first token is masked then index = 0

	Returns:
		A 1-D tensor of length equal to vocabulary where element in i'th index of tensor is the probability of i'th token 
		being a good replacement for masked token   
	
	'''
	

	out = masked_lm(**sample)

	logits = out.logits
	probs = F.softmax(logits, dim = -1) ##convert logits to probabilities

	assert np.isclose(sum(probs[0,0,:]).item(), 1)

	required_probabilities = probs[0,index,:]

	return required_probabilities

def calculate_target_class_prob(model: transformers.PreTrainedModel, data_loader: DataLoader, target_class: int, index: int) -> Tensor:
	
	target_class_probabilities = []
	
	model = model.cuda()
	with torch.no_grad():
		for batch in tqdm(data_loader, desc = f'calculating target class probs for index {index}'):
			for k,v in batch.items():
				batch[k] = v.cuda()

			out = model(**batch) #output logits are of shape (batch_size, num_classes) 

			probs = F.softmax(out.logits, dim = -1)
			assert np.isclose(sum(probs[0, :]).item(), 1)

			target_class_probabilities.append(probs[:, target_class])


	target_class_probabilities = torch.cat(target_class_probabilities, dim = -1).cpu()

	assert target_class_probabilities.ndim == 1
	assert len(target_class_probabilities) == len(data_loader.dataset)

	return target_class_probabilities


def calculate_log_odds(prob: float):
	assert prob <= 1 
    
	odds = prob/(1-prob)
	return np.log2(odds)


def marginalize_single_index(inp: Dict[str, Tensor], 
			     tokenizer: transformers.PreTrainedTokenizer, 
			     index: int, 
			     threshold: float, 
			     target_model: transformers.PreTrainedModel, 
			     language_model: transformers.PreTrainedModel, 
			     target_class: int, 
			     ) -> float:

	'''
	This function basically replaces the `index` in input with a random word in vocab and calculates 
	how much did the change affect the original output probability. 
	
	'''
	
	inp_length = inp['input_ids'].size(-1)
	replaced_word_id = inp['input_ids'][0, index]
	
	input_ids = inp['input_ids'].clone()
	input_ids[0, index] =  tokenizer.mask_token_id ##replace i'th word_id with [MASK] token_id which is 103 in BertTokenizer
			
			##prepare masked input to be passed through a masked language model
	masked_sample = {'input_ids': input_ids, 
			'attention_mask': inp['attention_mask'], 
			'token_type_ids': inp['token_type_ids']
			}

	## get the probability of other tokens appearing in place of masked token
	masked_probabilities = calculate_masked_probabilities(language_model, 
						masked_sample, 
						index = index, 
						)

	replaced_inputs = [] ## list of all the inputs after replacing the original token with other tokens
	masked_prob_gt_thresholds = []  ##list to collect probabilities of other tokens which are greater than thresholds
	eligible_words = 0 ## count of tokens eligible to replace original tokens
			
	for word,word_id in tokenizer.vocab.items(): ## loop over each word in vocabulary
		if '[' not in word and word_id != replaced_word_id:  ##avoid replacing by the word itself or special tokens.
			masked_prob = masked_probabilities[word_id]

			if masked_prob > threshold: ##avoid words with low likelihood
			
				replacable_input_ids = inp['input_ids'].clone() ##create a new clone each time
				replacable_input_ids[0,index] = word_id ##replace i'th word with other word in vocabulary
				
				replaced_inputs.append(replacable_input_ids)
				masked_prob_gt_thresholds.append(masked_prob)
				eligible_words += 1

	if len(replaced_inputs) == 0: ##No other token was good enough to replace the orginal token.
		return 1e-08

	masked_prob_gt_thresholds = torch.tensor(masked_prob_gt_thresholds)

	replaced_input_ids = torch.cat(replaced_inputs, dim = 0) ##right now replaced_input_ids are of shape (eligible_words,1,inp_length)
	attention_masks = inp['attention_mask'].repeat(eligible_words, 1)
	token_type_ids = inp['token_type_ids'].repeat(eligible_words, 1)

	assert tuple(replaced_input_ids.size()) == tuple(attention_masks.size()) == tuple(token_type_ids.size()) == (eligible_words, inp_length)

	ds = CustomDataset(replaced_input_ids, attention_masks, token_type_ids)
	dl = DataLoader(ds, batch_size= 32, sampler= SequentialSampler(ds))
				
	replaced_probs = calculate_target_class_prob(target_model,
						dl,
						target_class,
						index = index
						) ##calculating target class' probability with replaced word
	
	m = replaced_probs @ masked_prob_gt_thresholds

	total_masked_prob = sum(masked_prob_gt_thresholds)

	m /= total_masked_prob ##normalising the probabilities

	return m.item()



def calculate_input_marginalisation(target_model: transformers.PreTrainedModel,
				    language_model: transformers.PreTrainedModel, 
				    input_sentence: Union[str, List[str]], 
				    tokenizer: transformers.PreTrainedTokenizer, 
				    target_class: int, 
				    threshold:float = 1e-5,
				    ) -> Dict[str, Tensor]:

	# target_model = target_model.to('cuda')
	# language_model = language_model.to('cuda')

	target_model.eval()
	language_model.eval()

	if isinstance(input_sentence, List):
	    	inp = tokenizer(*input_sentence, return_tenors = 'pt')
	else:
		inp = tokenizer(input_sentence, return_tensors= 'pt')
		
	out = target_model(**inp)
	predicted_label = torch.argmax(out.logits, dim = -1).item()
	predicted_probs =  F.softmax(out.logits, dim = -1)
	true_class_probability = predicted_probs[0,target_class].item()

	confidence_in_predicted_label = predicted_probs[0, predicted_label].item()

	seq_length = inp['input_ids'].size(-1)
	original_sentence_tokenized = tokenizer.convert_ids_to_tokens(inp['input_ids'][0]) ## we do this to include special tokens that may have been introduced by the tokenizer
	assert len(original_sentence_tokenized) == seq_length 

	attribution_scores = {}
	attribution_scores_compiled = {}
	m_dict = []
    
	for i in range(seq_length): ##loop over each word token id
		if original_sentence_tokenized[i] == '[SEP]' or original_sentence_tokenized[i] == '[CLS]':
			continue
		m = marginalize_single_index(inp, tokenizer, i, threshold, target_model, language_model, target_class)
		m_dict.append((original_sentence_tokenized[i], m)) 
		attribution_scores[original_sentence_tokenized[i] + f'_{i}'] = calculate_log_odds(true_class_probability) - calculate_log_odds(m)


	items,probs = tuple(zip(*m_dict))
	indexes = list(range(len(items)))
	##The following loop calculates marginalization scores for words by combining the scores of all the tokens in which the word was divided 
	# for.eg word 'admirable' is broken into 'ad', '##mir', '##able' by wordpiece tokenizer. So to calculate the score for admirable we add the 
	# scores of it's tokens and divide by number of tokens 

	for index in indexes:
		word = items[index]
		prob = probs[index]
		
		k = 0
		for i,j in enumerate(items[index+1:]):
			if '##' in j:
				word += j.strip('#')
				prob += probs[index + i + 1]
				indexes.remove(index + i + 1)
				k += 1
			else:
				break
		prob /= (k+1)
		## + index to prevent same words in different index to collide
		attribution_scores_compiled[word + f'_{index}'] = calculate_log_odds(true_class_probability) - calculate_log_odds(prob)


	return attribution_scores_compiled,attribution_scores, m_dict,predicted_label,confidence_in_predicted_label
    