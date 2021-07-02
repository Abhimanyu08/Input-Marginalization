import argparse
from transformers import BertTokenizer
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

	args = parser.parse_args()

	target_model = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-SST-2')
	language_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	attribution_scores, _, predicted_label, confidence = calculate_input_marginalisation(target_model,
							language_model,
							args.sent,
							tokenizer,
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
						

