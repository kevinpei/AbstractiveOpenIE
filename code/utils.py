from scipy.special import softmax
from sentence_transformers import CrossEncoder
import pandas as pd
from datasets import Dataset

entailment_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

entailment_label_mapping = {'contradiction': 0, 'neutral': 2, 'entail': 1}

model_checkpoints = 't5-base'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoints)
tokenizer.add_tokens(['[arg1]', '[pred]', '[arg2]', '[et]'])



def preprocess_data(df):
	#get the text
	inputs = [text for text in df['text']]
	#tokenize text
	model_inputs = tokenizer(inputs,max_length=max_input, padding='max_length', truncation=True)

	#tokenize labels
	with tokenizer.as_target_tokenizer():
		targets = tokenizer(df['gold'], max_length=max_target, padding='max_length', truncation=True)
		
	model_inputs['labels'] = targets['input_ids']
	#returns input_ids, attention_masks, labels
	return model_inputs



def create_abstractive_file(filename, out_file, main_args):
	sentences = {}
	with open(filename, 'r') as f:
		for line in f:
			splitLine = line.strip().split('\t')
			sentence = splitLine[0]
			if len(splitLine) > 1:
				predicate = splitLine[1]
			if len(splitLine) > 2:
				arg1 = splitLine[2]
			if len(splitLine) > 3:
				arg2 = ' '.join(splitLine[3:])
			if len(predicate):
				relation = {'arg1': arg1, 'pred': predicate, 'arg2': arg2}
				if sentence not in sentences:
					sentences[sentence] = []
				sentences[sentence].append(relation)
	with open(out_file, 'w') as f:
		for sentence in sentences:
			pred_sentence = sentence
			pred_gold = ''
			for relation in sentences[sentence]:
				pred_gold += ' [pred] {}'.format(relation['pred'])
				f.write(main_args.arg_prompt.replace('<sent>', sentence).replace('<pred>', pred_gold) + '\t[arg1] {} [arg2] {}\n'.format(relation['arg1'], relation['arg2']))
			f.write(main_args.predicate_prompt.replace('<sent>', sentence) + '\t{}\n'.format(pred_gold.strip()))



def create_dataset(filename, main_args):
	df = pd.read_csv(filename, sep='\t', header=None, index_col=None)
	df.columns = ['text', 'gold']
	data = Dataset.from_pandas(df)
	tokenize_data = data.map(preprocess_data, batched = True, remove_columns=['text', 'gold'])
	return tokenize_data.shuffle(seed=main_args.seed)



def calculate_f1(p, r):
	return 2 * (float(p) * r) / (p + r)



def compute_tuple_tuple_score(pred):
	predictions, labels = pred
	#decode the predictions
	decode_predictions = [prediction.replace('[arg1]', '').replace('[pred]', '').replace('[arg2]', '').replace('[et]', '') for prediction in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
	#decode labels
	decode_labels = [label.replace('[arg1]', '').replace('[pred]', '').replace('[arg2]', '').replace('[et]', '') for label in tokenizer.batch_decode(labels, skip_special_tokens=True)]

	precision_examples = [(prediction, label) for prediction, label in zip(decode_predictions, decode_labels)]

	prec_scores = entailment_model.predict(precision_examples)
	prec_probs = softmax(prec_scores, axis=1)
	prec_probs = [prec_prob[entailment_label_mapping['entail']] for prec_prob in prec_probs]

	recall_examples = [(label, prediction) for label, prediction in zip(decode_labels, decode_predictions)]

	recall_scores = entailment_model.predict(recall_examples)
	recall_probs = softmax(recall_scores, axis=1)
	recall_probs = [recall_prob[entailment_label_mapping['entail']] for recall_prob in recall_probs]

	f1_probs = [calculate_f1(p, r) for p, r in zip(prec_probs, recall_probs)]

	metric_dict = {'precision': prec_probs, 'recall': recall_probs, 'f1_probs': f1_probs, 'hashcode': ''}

	return metric_dict