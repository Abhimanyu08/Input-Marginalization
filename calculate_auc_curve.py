from typing import Dict, List,Union
from torch.utils.data import DataLoader, SequentialSampler
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer
from utils import CustomDataset
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch.nn as nn

def calculate_masked_probs(data_loader: DataLoader,
			   model: nn.Module,
			  device:str = 'cuda') -> Tensor:
	outputs = []
	model.eval()
	model = model.to(device)
	with torch.no_grad():
		for batch in tqdm(data_loader, desc = 'calculating probability distributions after masking'):
			for i,j in batch.items():
				batch[i] = j.to(device)

			out = model(**batch)

			logits = out.logits ##logits are of shape  (batch_size, seq_len, vocab_size)
			probabilities = F.softmax(logits, dim = -1)
			outputs.append(probabilities.cpu())

	model = model.to('cpu')
	outputs = torch.cat(outputs, dim = 0)
	assert outputs.size(0) == len(data_loader.dataset)
	return outputs 

def calculate_target_class_probabilities(data_loader: DataLoader,
					model: nn.Module,
					class_label: int, 
					device = 'cuda',
					) -> Tensor:

	model = model.eval()
	model = model.to(device)
	outputs = []

	with torch.no_grad():
		for batch in tqdm(data_loader, desc= 'calculating cofidence after replacing by random words'):
			for i,j in batch.items():
				batch[i] = j.to(device)

			out = model(**batch)

			logits = out.logits ##logits are of shape (batch_size, num_classes)
			probs = F.softmax(logits, dim = -1)
			outputs.append(probs[:, class_label])

	outputs = torch.cat(outputs, dim = -1).cpu()
	model = model.to('cpu')

	assert outputs.ndim == 1
	print(outputs.size())
	assert outputs.size() == (len(data_loader.dataset),)

	return outputs


def calculate_confidence_scores(sentence: Union[str, List[str]],
			attribution_scores: Dict[str, float],
			tokenizer: PreTrainedTokenizer,
			language_model: nn.Module,
			target_model: nn.Module,
			true_class_label:int,
			true_class_confidence: float,
			replace_randomly:bool = False,
			device:str = 'cuda',
			):

	###Preprocess sentence
	tokenizer_out = tokenizer(sentence, return_tensors= 'pt') if isinstance(sentence, str) else tokenizer(**sentence, return_tensors = 'pt')	
	tokenized_input = tokenizer.convert_ids_to_tokens(list(tokenizer_out['input_ids'][0]))

	### Sort the tokens in decreasing order of attribution scores
	inp_without_special_tokens = [inp + f'_{i}' for i,inp in enumerate(tokenized_input) if '[' not in inp]
	sorted_input  = sorted(inp_without_special_tokens, key = lambda x: attribution_scores[x], reverse= True) ##sort the inputs in decreasing order of attribution scores


	input_ids = tokenizer_out['input_ids']  ##input_ids are of shape (1,seq_length) 
	num_inputs_to_replace = int(0.2*len(tokenized_input)) ## we are only allowed to replace 20% of the tokens


	### Replace the 20% tokens with mask tokens
	truncated_inputs = []
	num_replaced = 0
	cloned_ids =  input_ids.clone()
	for token in sorted_input:

		_, index = token.split('_')

		cloned_ids[0,int(index)] = tokenizer.mask_token_id		
		num_replaced += 1

		truncated_inputs.append(cloned_ids.clone())
		if num_replaced + 1 > num_inputs_to_replace:
			break

	### pass them through a MLM to find the probability distribution over the masked tokens
	mlm_input_ids = torch.cat(truncated_inputs, dim = 0)
	attention_mask = tokenizer_out['attention_mask'].repeat(num_replaced, 1)
	token_type_ids = tokenizer_out['token_type_ids'].repeat(num_replaced, 1)

	assert tuple(mlm_input_ids.size()) == tuple(attention_mask.size()) == tuple(token_type_ids.size()) == (num_replaced, len(tokenized_input))

	masked_ds = CustomDataset(input_ids = mlm_input_ids, attention_masks = attention_mask, token_type_ids = token_type_ids)
	mlm_dl = DataLoader(masked_ds, batch_size = 16, sampler = SequentialSampler(masked_ds))

	probability_distributions = calculate_masked_probs(mlm_dl, language_model, device = device) ## probability distributions should be of shape (num_replaced, input_len, vocab_size)
	
	assert probability_distributions.size() == (num_replaced, len(tokenized_input), tokenizer.vocab_size)
	assert torch.allclose(probability_distributions.sum(dim = -1), torch.ones(num_replaced, len(tokenized_input)))
	

	masked_indexes = mlm_input_ids == tokenizer.mask_token_id
	masked_indexes = list(masked_indexes.nonzero())
	replaced_input_ids = mlm_input_ids.clone()

	## replace the tokens with high attribution scores with other tokens and calculate the confidence in predicted label by the said change.
	for index in masked_indexes:
		index= tuple(index)
		id_to_replace = input_ids[0, index[1]]
		if replace_randomly:
			id_to_replace_by = id_to_replace
			while id_to_replace_by == id_to_replace:
				id_to_replace_by = torch.multinomial(probability_distributions[index], num_samples = 1).squeeze_()
		else:
			top2 = torch.topk(probability_distributions[index], k =2).indices
			id_to_replace_by = top2[0] if top2[0] != id_to_replace else top2[1]
		replaced_input_ids.index_put_(indices = index, values = id_to_replace_by)

	assert replaced_input_ids.size() == (num_replaced, len(tokenized_input))

	replaced_ds = CustomDataset(replaced_input_ids, attention_mask, token_type_ids)
	replaced_dl = DataLoader(replaced_ds, batch_size= 16, sampler = SequentialSampler(replaced_ds))

	replaced_probs = calculate_target_class_probabilities(replaced_dl, target_model, true_class_label, device =device)

	assert replaced_probs.size() == (num_replaced,)

	confidence_dict = {0: true_class_confidence}

	for num_replaced,prob in zip(range(1,num_replaced + 1), replaced_probs):
		confidence_dict[num_replaced] = prob.item()

	return confidence_dict


def calculate_auc_rep(confidence_scores: Dict[int, float]):
	x = np.array(list(confidence_scores.keys()))
	y = np.array(list(confidence_scores.values()))

	return np.trapz(y, x)	






	
	


	




	

	
