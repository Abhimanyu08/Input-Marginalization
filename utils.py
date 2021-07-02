from typing import Union, List, Dict
import transformers
from datasets import load_dataset, Dataset

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