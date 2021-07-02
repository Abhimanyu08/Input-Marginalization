import argparse
from transformers import BertTokenizer
import transformers
from input_marginalization import calculate_input_marginalisation
from transformers import BertForMaskedLM, BertForSequenceClassification 
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--sent', type = str, help = "sentence for which you want to calculate the marginalization scores")
	parser.add_argument('--label', type = int, default= 1, 
				help = "optional label for what sentiment you think the sentence conveys, 0 of negative, 1 for positive")

	parser.add_argument('--target_model', 
				type = transformers.PreTrainedModel,
				default= BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-SST-2'),
				help = 'model whose predictions you want to interpret. Model class should be inherited from `transformers.PreTrainedModel`. Default = `BertForSequenceClassification.from_pretrained(\'textattack/bert-base-uncased-SST-2\')`')

	parser.add_argument('--masked_language_model', 
				type = transformers.PreTrainedModel,
				default= BertForMaskedLM.from_pretrained('bert-base-uncased'),
				help = "Language Model to be used to in input marginalization. Should be inherited from class `transformers.PreTrainedModel`.Default = `BertForSequenceClassification.from_pretrained(\'bert-base-uncased\')`")

	parser.add_argument('--tokenizer', 
				type = transformers.PreTrainedTokenizer,
				default= BertTokenizer.from_pretrained('bert-base-uncased'),
				help = "Tokenizer to use to preproces the input sentence. Should be inherited from `transformers.PreTrainedTokenizer`. Default = BertTokenizer.from_pretrained('bert-base-uncased')")
	

	args = parser.parse_args()


	attribution_scores, _, predicted_label, confidence = calculate_input_marginalisation(args.target_model,
							args.language_model,
							args.sent,
							args.tokenizer,
							args.label)


	print(attribution_scores)
	print(f'Correct label = {args.label} \n Predicted Label = {predicted_label} with confidence {confidence}')
	words = list(attribution_scores.keys())

	# words = [i.split('_')[0] for i in words]
	scores = list(attribution_scores.values())
	
	fig = plt.figure(figsize = (10, 5))
	
	# creating the bar plot
	plt.bar(words,scores, color ='red',
		width = 0.4)
	
	plt.xlabel("Words")
	plt.ylabel("Attribution Scores")
	plt.title("Scores for each word")

	plt.show()

if __name__ == '__main__':
	main()	
						

