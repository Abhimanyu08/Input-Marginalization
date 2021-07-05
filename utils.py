from typing import Union, List, Dict
import transformers
from datasets import load_dataset, Dataset

class CustomDataset():
	def __init__(self, input_ids, attention_masks, token_type_ids):
		self.input_ids = input_ids
		self.attention_masks = attention_masks
		self.token_type_ids = token_type_ids

	def __len__(self): return self.input_ids.size(0)
	

	def __getitem__(self, i): return {'input_ids': self.input_ids[i,:], 'attention_mask': self.attention_masks[i,:], 'token_type_ids': self.token_type_ids[i, :]}


def prepare_dataset(name: str, 
		    tokenizer: transformers.PreTrainedTokenizer,
		    split: Union[List[str]] = ['train', 'validation'],
		    max_length = 128) -> Dict[str, Dataset]:
    
	dataset = load_dataset('glue', name, split = split)

	text_features= []

	for key,value in dataset[0].features.items():
		if value.dtype == 'string':
			text_features.append(key)
		
	dataset_dict = {}

	for ds,split_name in zip(dataset, split):
		ds = ds.map(lambda examples: tokenizer(*(examples[i] for i in text_features), padding = 'max_length', max_length = max_length, truncation = True), batched = True)

		ds = ds.map(lambda examples : {'labels': examples['label']}, batched = True)

		ds.set_format(type = 'torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

		dataset_dict[split_name] = ds

	return dataset_dict