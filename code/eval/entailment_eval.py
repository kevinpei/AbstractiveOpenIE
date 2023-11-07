from bert_nli import BertNLIModel
import collections
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from scipy.special import softmax
import argparse
import copy

advanced_bert_type = 'cross-encoder/nli-deberta-v3-base'
sota_model = CrossEncoder(advanced_bert_type)


def read_extraction_file(filename, is_prediction, predictedsent2goldsent=None):
	sent2extraction = {}
	with open(filename, 'r') as f:
		for line in f:
			splitLine = line.strip().split('\t')
			if predictedsent2goldsent:
				if splitLine[0] in predictedsent2goldsent:
					sentence = predictedsent2goldsent[splitLine[0]]
				else:
					sentence = splitLine[0]
			else:
				sentence = splitLine[0]
			predicate = ''
			arg1 = ''
			arg2s = ['']
			if len(splitLine) > 1+is_prediction:
				predicate = splitLine[1+is_prediction]
			if len(splitLine) > 2+is_prediction:
				arg1 = splitLine[2+is_prediction]
			if len(splitLine) > 3+is_prediction:
				if is_prediction:
					arg2s = splitLine[4:7]
				else:
					arg2s = splitLine[3:]
			extraction = '{} {} {}'.format(arg1, predicate, ' '.join(arg2s))
			if sentence not in sent2extraction:
				sent2extraction[sentence] = []
			sent2extraction[sentence].append(extraction)
	return sent2extraction


def read_sentence_file(filename):
	sents = []
	with open(filename, 'r') as f:
		for line in f:
			sents.append(line.strip())
	return sents


def read_predictedsent2goldsent(filename):
	mapping = {}
	with open(filename, 'r') as f:
		for line in f:
			splitLine = line.strip().split('\t')
			mapping[splitLine[1]] = splitLine[0]
	return mapping


label2prob = {'contradiction': 0, 'neutral': 2, 'entail': 1}
label_mapping = ['contradiction', 'entailment', 'neutral']


def calculate_f1(p, r):
	return 2 * (float(p) * r) / (p + r)


def combine_tuples(tuple_list):
	combined_tuples = ''
	for extraction in tuple_list:
		if len(extraction.strip()):
			combined_tuples += extraction + '. '
	return combined_tuples.strip()


def calculate_combined_tuple_tuple_score(sent2gold, sent2predictions, prediction_name, information_dict, model, sent_level_out_file):
	precision_examples = []
	id2sent = {}
	example_id = 0
	for sentence in sent2predictions:
		if sentence in sent2gold:
			combined_gold = combine_tuples(sent2gold[sentence])
		else:
			continue
		for predicted_tuple in sent2predictions[sentence]:
			if not len(predicted_tuple):
				continue
			precision_examples.append((combined_gold, predicted_tuple))
			id2sent[example_id] = sentence
			example_id += 1

	#labels, probs = model(precision_examples)
	scores = model.predict(precision_examples)
	probs = softmax(scores, axis=1)
	labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]

	sent2precision_entailment = {}
	for i in range(len(labels)):
		sentence = id2sent[i]
		if sentence not in sent2precision_entailment:
			sent2precision_entailment[sentence] = []
		sent2precision_entailment[sentence].append(probs[i][label2prob['entail']])

	recall_examples = []
	id2sent = {}
	example_id = 0
	for sentence in sent2gold:
		if sentence in sent2predictions:
			combined_predictions = combine_tuples(sent2predictions[sentence])
		else:
			#combined_predictions = 'NA.'
			continue
		if not len(combined_predictions):
			continue
		for gold_tuple in sent2gold[sentence]:
			recall_examples.append((combined_predictions, gold_tuple))
			id2sent[example_id] = sentence
			example_id += 1

	try:
		#labels, probs = model(recall_examples)
		scores = model.predict(recall_examples)
		probs = softmax(scores, axis=1)
		labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
	except TypeError:
		labels, probs = ([0], [[0, 0, 0]])

	sent2recall_entailment = {}
	for i in range(len(labels)):
		sentence = id2sent[i]
		if sentence not in sent2recall_entailment:
			sent2recall_entailment[sentence] = []
		sent2recall_entailment[sentence].append(probs[i][label2prob['entail']])

	print("Finished combined tuple to tuple")

	recalls = {}
	precisions = {}
	f1s = {}

	for sent in sent2recall_entailment.keys():
		if sent not in information_dict:
			information_dict[sent] = {}
		information_dict[sent]['combined_tuple_r'] = np.average(sent2recall_entailment[sent])
		information_dict[sent]['combined_tuple_individual_prediction_r'] = sent2recall_entailment[sent]
		recalls[sent] = np.average(sent2recall_entailment[sent])
	for sent in sent2precision_entailment.keys():
		if sent not in information_dict:
			information_dict[sent] = {}
		information_dict[sent]['combined_tuple_p'] = np.average(sent2precision_entailment[sent])
		information_dict[sent]['combined_tuple_individual_prediction_p'] = sent2precision_entailment[sent]
		precisions[sent] = np.average(sent2precision_entailment[sent])
	for sent in precisions.keys():
		if sent in recalls:
			f1s[sent] = calculate_f1(precisions[sent], recalls[sent])
			information_dict[sent]['combined_tuple_f1'] = calculate_f1(precisions[sent], recalls[sent])
			information_dict[sent]['combined_tuple_individual_prediction_f1'] = [calculate_f1(p, r) for p, r in zip(information_dict[sent]['combined_tuple_individual_prediction_p'], information_dict[sent]['combined_tuple_individual_prediction_r'])]
		else:
			f1s[sent] = calculate_f1(precisions[sent], 0)
			recalls[sent] = 0
			information_dict[sent]['combined_tuple_r'] = 0
			information_dict[sent]['combined_tuple_individual_prediction_r'] = [0]
			information_dict[sent]['combined_tuple_f1'] = calculate_f1(precisions[sent], 0)
			information_dict[sent]['combined_tuple_individual_prediction_f1'] = [calculate_f1(p, 0) for p in information_dict[sent]['combined_tuple_individual_prediction_p']]
	for sent in recalls.keys():
		if sent in precisions:
			if sent not in f1s:
				f1s[sent] = calculate_f1(precisions[sent], recalls[sent])
				information_dict[sent]['combined_tuple_f1'] = calculate_f1(precisions[sent], recalls[sent])
				information_dict[sent]['combined_tuple_individual_prediction_f1'] = [calculate_f1(p, r) for p, r in zip(information_dict[sent]['combined_tuple_individual_prediction_p'], information_dict[sent]['combined_tuple_individual_prediction_r'])]
		else:
			if sent not in f1s:
				f1s[sent] = calculate_f1(0, recalls[sent])
				information_dict[sent]['combined_tuple_f1'] = calculate_f1(0, recalls[sent])
				information_dict[sent]['combined_tuple_individual_prediction_f1'] = [calculate_f1(0, r) for r in zip(information_dict[sent]['combined_tuple_individual_prediction_r'])]
			precisions[sent] = 1.0
			information_dict[sent]['combined_tuple_p'] = 1.0
			information_dict[sent]['combined_tuple_individual_prediction_p'] = [1.0]

	#p, r, f1 = np.average(list(precisions.values())), np.average(list(recalls.values())), np.average(list(f1s.values()))
	p, r = np.average(list(precisions.values())), np.average(list(recalls.values()))
	f1 = calculate_f1(p, r)

	with open(sent_level_out_file, 'a') as f:
		f.write('{}\t{}\t{}\t{}\n'.format(prediction_name, p, r, f1))

	return p, r, f1


def calculate_sentence_tuple_score(sent2gold, sent2predictions, prediction_name, information_dict, model, sent_level_out_file):
	precision_examples = []
	id2sent = {}
	example_id = 0
	for sentence in sent2predictions:
		if sentence in sent2gold:
			combined_predictions = combine_tuples(sent2predictions[sentence])
			precision_examples.append((sentence, combined_predictions))
			id2sent[example_id] = sentence
			example_id += 1

	try:
		#labels, probs = model(precision_examples)
		scores = model.predict(precision_examples)
		probs = softmax(scores, axis=1)
		labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
	except TypeError:
		labels, probs = ([0], [[0, 0, 0]])

	sent2precision_entailment = {}
	for i in range(len(labels)):
		sentence = id2sent[i]
		if sentence not in sent2precision_entailment:
			sent2precision_entailment[sentence] = []
		sent2precision_entailment[sentence].append(probs[i][label2prob['entail']])

	recall_examples = []
	id2sent = {}
	example_id = 0
	for sentence in sent2gold:
		if sentence in sent2predictions:
			combined_predictions = combine_tuples(sent2predictions[sentence])
			recall_examples.append((combined_predictions, sentence))
			id2sent[example_id] = sentence
			example_id += 1
		else:
			recall_examples.append(('N/A', sentence))
			id2sent[example_id] = sentence
			example_id += 1
	try:
		#labels, probs = model(recall_examples)
		scores = model.predict(recall_examples)
		probs = softmax(scores, axis=1)
		labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
	except TypeError:
		labels, probs = ([0], [[0, 0, 0]])

	sent2recall_entailment = {}
	for i in range(len(labels)):
		sentence = id2sent[i]
		if sentence not in sent2recall_entailment:
			sent2recall_entailment[sentence] = []
		sent2recall_entailment[sentence].append(probs[i][label2prob['entail']])

	print("Finished sentence to tuple")

	recalls = {}
	precisions = {}
	f1s = {}

	for sent in sent2recall_entailment.keys():
		recalls[sent] = np.average(sent2recall_entailment[sent])
		if sent not in information_dict:
			information_dict[sent] = {}
		information_dict[sent]['sentence_r'] = np.average(sent2recall_entailment[sent])
		information_dict[sent]['sentence_individual_prediction_r'] = sent2recall_entailment[sent]
	for sent in sent2precision_entailment.keys():
		precisions[sent] = np.average(sent2precision_entailment[sent])
		if sent not in information_dict:
			information_dict[sent] = {}
		information_dict[sent]['sentence_p'] = np.average(sent2precision_entailment[sent])
		information_dict[sent]['sentence_individual_prediction_p'] = sent2precision_entailment[sent]
	for sent in precisions.keys():
		if sent in recalls:
			f1s[sent] = calculate_f1(precisions[sent], recalls[sent])
			information_dict[sent]['sentence_f1'] = calculate_f1(precisions[sent], recalls[sent])
			information_dict[sent]['sentence_individual_prediction_f1'] = [calculate_f1(p, r) for p, r in zip(information_dict[sent]['sentence_individual_prediction_p'], information_dict[sent]['sentence_individual_prediction_r'])]
		else:
			f1s[sent] = calculate_f1(precisions[sent], 0)
			recalls[sent] = 0
			information_dict[sent]['sentence_r'] = 0
			information_dict[sent]['sentence_individual_prediction_r'] = [0]
			information_dict[sent]['sentence_f1'] = calculate_f1(precisions[sent], 0)
			information_dict[sent]['sentence_individual_prediction_f1'] = [calculate_f1(p, 0) for p in information_dict[sent]['sentence_individual_prediction_p']]
	for sent in recalls.keys():
		if sent in precisions:
			if sent not in f1s:
				f1s[sent] = calculate_f1(precisions[sent], recalls[sent])
				information_dict[sent]['sentence_f1'] = calculate_f1(precisions[sent], recalls[sent])
				information_dict[sent]['sentence_individual_prediction_f1'] = [calculate_f1(p, r) for p, r in zip(information_dict[sent]['sentence_individual_prediction_p'], information_dict[sent]['sentence_individual_prediction_r'])]
		else:
			if sent not in f1s:
				f1s[sent] = calculate_f1(0, recalls[sent])
				information_dict[sent]['sentence_f1'] = calculate_f1(0, recalls[sent])
				information_dict[sent]['sentence_individual_prediction_f1'] = [calculate_f1(0, r) for r in information_dict[sent]['sentence_individual_prediction_r']]
			precisions[sent] = 1.0
			information_dict[sent]['sentence_p'] = 1.0
			information_dict[sent]['sentence_individual_prediction_p'] = [1.0]

	#p, r, f1 = np.average(list(precisions.values())), np.average(list(recalls.values())), np.average(list(f1s.values()))
	p, r = np.average(list(precisions.values())), np.average(list(recalls.values()))
	f1 = calculate_f1(p, r)

	with open(sent_level_out_file, 'a') as f:
		f.write('{}\t{}\t{}\t{}\n'.format(prediction_name, p, r, f1))

	return p, r, f1


def calculate_tuple_tuple_score(sent2gold, sent2predictions, prediction_name, information_dict, model, sent_level_out_file):
	precision_examples = []
	id2sent = {}
	sent2id = {}
	example_id = 0

	unmatched_precision_predictions = {}
	for sent in sent2gold:
		unmatched_precision_predictions[sent] = []
		for gold_tuple in sent2gold[sent]:
			unmatched_precision_predictions[sent].append(gold_tuple)

	for sentence in sent2predictions:
		for predicted_tuple in sent2predictions[sentence]:
			if not len(predicted_tuple):
				continue
			if sentence in sent2gold:
				for gold_tuple in sent2gold[sentence]:
					precision_examples.append((gold_tuple, predicted_tuple))
					id2sent[example_id] = sentence
					if sentence not in sent2id:
						sent2id[sentence] = []
					sent2id[sentence].append(example_id)
					example_id += 1

	#labels, probs = model(precision_examples)
	scores = model.predict(precision_examples)
	probs = softmax(scores, axis=1)
	labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
		
	sent2precision_entailment = {}
	matched_precision_predictions = {}
	for sent in sent2id:
		scores = {}
		predictions = set()
		for example_id in sent2id[sent]:
			gold, prediction = precision_examples[example_id]
			predictions.add(prediction)
			if gold not in scores:
				scores[gold] = {}
			scores[gold][prediction] = probs[example_id][label2prob['entail']]

		selected_rows = []
		selected_cols = []
		num_precision_matches = min(len(scores), len(predictions))
		precision_numerator = 0
		for t in range(num_precision_matches):
			matched_row = -1
			matched_col = -1
			matched_precision = -np.inf # initialised to <0 so that it updates whenever precision is 0 as well
			for gold in scores:
				if gold in selected_rows:
					continue
				for prediction in predictions:
					if prediction in selected_cols:
						continue
					if scores[gold][prediction] > matched_precision:
						matched_precision = scores[gold][prediction]
						matched_row = gold
						matched_col = prediction

			selected_rows.append(matched_row)
			selected_cols.append(matched_col)
			if sent not in matched_precision_predictions:
				matched_precision_predictions[sent] = []
			#matched_precision_predictions[sent][matched_row] = matched_col
			matched_precision_predictions[sent].append((matched_row, matched_col))
			unmatched_precision_predictions[sent].remove(matched_row)
			#precision_numerator += scores[matched_row][matched_col]
			if sent not in sent2precision_entailment:
				sent2precision_entailment[sent] = []
			sent2precision_entailment[sent].append(scores[matched_row][matched_col])

	recall_examples = []
	id2sent = {}
	sent2id = {}
	example_id = 0
	for sentence in sent2gold:
		for gold_tuple in sent2gold[sentence]:
			if sentence in sent2predictions:
				for predicted_tuple in sent2predictions[sentence]:
					if not len(predicted_tuple):
						continue
					recall_examples.append((predicted_tuple, gold_tuple))
					id2sent[example_id] = sentence
					if sentence not in sent2id:
						sent2id[sentence] = []
					sent2id[sentence].append(example_id)
					example_id += 1
			else:
				recall_examples.append((gold_tuple, ''))
				id2sent[example_id] = sentence
				if sentence not in sent2id:
					sent2id[sentence] = []
				sent2id[sentence].append(example_id)
				example_id += 1

	#labels, probs = model(recall_examples)
	scores = model.predict(recall_examples)
	probs = softmax(scores, axis=1)
	labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
		
	sent2recall_entailment = {}
	matched_recall_predictions = {}
	for sent in sent2id:
		scores = {}
		if sent not in matched_recall_predictions:
			matched_recall_predictions[sent] = []
		for example_id in sent2id[sent]:
			prediction, gold = recall_examples[example_id]
			if gold not in scores:
				scores[gold] = []
			scores[gold].append((prediction, probs[example_id][label2prob['entail']]))
		if sent not in sent2recall_entailment:
			sent2recall_entailment[sent] = []
		for gold in scores:
			prediction, max_score = max(scores[gold], key=lambda x: x[1])
			sent2recall_entailment[sent].append(max_score)
			matched_recall_predictions[sent].append(prediction)

	print("Finished tuple to tuple")

	recalls = {}
	precisions = {}
	f1s = {}

	for sent in unmatched_precision_predictions:
		if sent not in information_dict:
			information_dict[sent] = {}
		information_dict[sent]['unmatched_precision_predictions'] = unmatched_precision_predictions[sent]

	for sent in sent2recall_entailment.keys():
		recalls[sent] = np.average(sent2recall_entailment[sent])
		if sent not in information_dict:
			information_dict[sent] = {}
		information_dict[sent]['tuple_r'] = np.average(sent2recall_entailment[sent])
		information_dict[sent]['tuple_individual_prediction_r'] = sent2recall_entailment[sent]
		information_dict[sent]['matched_recall_predictions'] = matched_recall_predictions[sent]
	for sent in sent2precision_entailment.keys():
		precisions[sent] = np.average(sent2precision_entailment[sent])
		if sent not in information_dict:
			information_dict[sent] = {}
		information_dict[sent]['tuple_p'] = np.average(sent2precision_entailment[sent])
		information_dict[sent]['tuple_individual_prediction_p'] = sent2precision_entailment[sent]
		information_dict[sent]['matched_precision_predictions'] = matched_precision_predictions[sent]
	for sent in precisions.keys():
		if sent in recalls:
			f1s[sent] = calculate_f1(precisions[sent], recalls[sent])
			information_dict[sent]['tuple_f1'] = calculate_f1(precisions[sent], recalls[sent])
			information_dict[sent]['tuple_individual_prediction_f1'] = [calculate_f1(p, r) for p, r in zip(information_dict[sent]['tuple_individual_prediction_p'], information_dict[sent]['tuple_individual_prediction_r'])]
		else:
			f1s[sent] = calculate_f1(precisions[sent], 0)
			recalls[sent] = 0
			information_dict[sent]['tuple_r'] = 0
			information_dict[sent]['tuple_individual_prediction_r'] = [0]
			information_dict[sent]['tuple_f1'] = calculate_f1(precisions[sent], 0)
			information_dict[sent]['tuple_individual_prediction_f1'] = [calculate_f1(p, 0) for p in information_dict[sent]['tuple_individual_prediction_p']]
	for sent in recalls.keys():
		if sent in precisions:
			if sent not in f1s:
				f1s[sent] = calculate_f1(precisions[sent], recalls[sent])
				information_dict[sent]['tuple_f1'] = calculate_f1(precisions[sent], recalls[sent])
				information_dict[sent]['tuple_individual_prediction_f1'] = [calculate_f1(p, r) for p, r in zip(information_dict[sent]['tuple_individual_prediction_p'], information_dict[sent]['tuple_individual_prediction_r'])]
		else:
			if sent not in f1s:
				f1s[sent] = calculate_f1(0, recalls[sent])
				information_dict[sent]['tuple_f1'] = calculate_f1(0, recalls[sent])
				information_dict[sent]['tuple_individual_prediction_f1'] = [calculate_f1(0, r) for r in information_dict[sent]['tuple_individual_prediction_r']]
			precisions[sent] = 1.0
			information_dict[sent]['tuple_p'] = 1.0
			information_dict[sent]['tuple_individual_prediction_p'] = [1.0]

	#p, r, f1 = np.average(list(precisions.values())), np.average(list(recalls.values())), np.average(list(f1s.values()))
	p, r = np.average(list(precisions.values())), np.average(list(recalls.values()))
	f1 = calculate_f1(p, r)

	with open(sent_level_out_file, 'a') as f:
		f.write('{}\t{}\t{}\t{}\n'.format(prediction_name, p, r, f1))

	return p, r, f1
	


def check_inferred(sent, gold_tuple):
	for token in gold_tuple.strip().split():
		if token not in sent:
			return 'True'
	return 'False'



def count_inferred(sent, information_dict):
	num_inferred = 0
	total_num = 0
	if 'matched_precision_predictions' in information_dict[sent]:
		for i, (gold_tuple, prediction) in enumerate(information_dict[sent]['matched_precision_predictions']):
			if check_inferred(sent, gold_tuple) == 'True':
				num_inferred += 1
			total_num += 1
		for i, gold_tuple in enumerate(information_dict[sent]['unmatched_precision_predictions']):
			if check_inferred(sent, gold_tuple) == 'True':
				num_inferred += 1
			total_num += 1
	else:
		for i, gold_tuple in enumerate(information_dict[sent]['unmatched_precision_predictions']):
			if check_inferred(sent, gold_tuple) == 'True':
				num_inferred += 1
			total_num += 1
	return num_inferred, total_num




def write_information_dict(information_dict, out_file):
	with open(out_file, 'w') as f:
		f.write('sentence\tnum_gold\tgold tuple\tis_inferred\tnum_inferred\tbest precision predicted tuple')
		for metric_type in ['sentence', 'combined_tuple', 'tuple']:
			for metric in ['p', 'r', 'f1']:
				f.write('\t{}_{}'.format(metric_type, metric))
		for metric_type in ['tuple']:
			for metric in ['p', 'r', 'f1']:
				f.write('\t{}_individual_prediction_{}'.format(metric_type, metric))

		average_inferred = {}
		average_uninferred = {}

		f.write('\n')
		for sent in information_dict:
			num_inferred, total_num = count_inferred(sent, information_dict)
			if 'matched_precision_predictions' in information_dict[sent]:
				for i, (gold_tuple, prediction) in enumerate(information_dict[sent]['matched_precision_predictions']):
					#f.write('{}\t{}\t{}\t'.format(sent, gold_tuple, information_dict[sent]['matched_precision_predictions'][gold_tuple]))
					is_inferred = check_inferred(sent, gold_tuple)
					f.write('{}\t{}\t{}\t{}\t{}\t{}\t'.format(sent, total_num, gold_tuple, is_inferred, num_inferred, prediction))
					for metric_type in ['sentence', 'combined_tuple', 'tuple']:
						for metric in ['p', 'r', 'f1']:
							combined_metric = '{}_{}'.format(metric_type, metric)
								
							if combined_metric in information_dict[sent]:
								f.write('{}\t'.format(information_dict[sent][combined_metric]))
								average_inferred[combined_metric] = [information_dict[sent][combined_metric]] * num_inferred
								average_uninferred[combined_metric] = [information_dict[sent][combined_metric]] * (total_num - num_inferred)
							else:
								if metric == 'p':
									f.write('{}\t'.format(1.0))
									average_inferred[combined_metric] = [1.0] * num_inferred
									average_uninferred[combined_metric] = [1.0] * (total_num - num_inferred)
								else:
									f.write('{}\t'.format(0.0))
									average_inferred[combined_metric] = [0.0] * num_inferred
									average_uninferred[combined_metric] = [0.0] * (total_num - num_inferred)

					for metric_type in ['tuple']:
						for metric in ['p', 'r', 'f1']:
							combined_metric = '{}_individual_prediction_{}'.format(metric_type, metric)
								
							if combined_metric in information_dict[sent]:
								f.write('{}\t'.format(information_dict[sent][combined_metric][i]))
							else:
								if metric == 'p':
									f.write('{}\t'.format(1.0))
								else:
									f.write('{}\t'.format(0.0))
					f.write('\n')

				for i, gold_tuple in enumerate(information_dict[sent]['unmatched_precision_predictions']):
					is_inferred = check_inferred(sent, gold_tuple)
					f.write('{}\t{}\t{}\t{}\t{}\t{}\t'.format(sent, total_num, gold_tuple, is_inferred, num_inferred, 'N/A'))
					for metric_type in ['sentence', 'combined_tuple', 'tuple']:
						for metric in ['p', 'r', 'f1']:
							combined_metric = '{}_{}'.format(metric_type, metric)
								
							if combined_metric in information_dict[sent]:
								f.write('{}\t'.format(information_dict[sent][combined_metric]))
							else:
								if metric == 'p':
									f.write('{}\t'.format(1.0))
									average_inferred[combined_metric] = [1.0] * num_inferred
									average_uninferred[combined_metric] = [1.0] * (total_num - num_inferred)
								else:
									f.write('{}\t'.format(0.0))
									average_inferred[combined_metric] = [0.0] * num_inferred
									average_uninferred[combined_metric] = [0.0] * (total_num - num_inferred)

					for metric_type in ['tuple']:
						f.write('1.0\t0.0\t0.0\t')
					f.write('\n')

			else:
				for i, gold_tuple in enumerate(information_dict[sent]['unmatched_precision_predictions']):
					is_inferred = check_inferred(sent, gold_tuple)
					f.write('{}\t{}\t{}\t{}\t{}\t{}\t'.format(sent, total_num, gold_tuple, is_inferred, num_inferred, 'N/A'))
					for metric_type in ['sentence', 'combined_tuple', 'tuple']:
						f.write('0.0\t0.0\t0.0\t')
					for metric_type in ['tuple']:
						f.write('0.0\t0.0\t0.0\t')
					f.write('\n')

		f.write('\n')
		for metric_type in ['sentence', 'combined_tuple', 'tuple']:
			for metric in ['f1']:
				f.write('average_inferred_{}_{}\t'.format(metric_type, metric))
		for metric_type in ['sentence', 'combined_tuple', 'tuple']:
			for metric in ['f1']:
				f.write('average_non_inferred_{}_{}\t'.format(metric_type, metric))
		f.write('\n')
		for metric_type in ['sentence', 'combined_tuple', 'tuple']:
			for metric in ['f1']:
				if metric in average_inferred:
					f.write('{}\t'.format(np.mean(average_inferred[metric])))
		for metric_type in ['sentence', 'combined_tuple', 'tuple']:
			for metric in ['f1']:
				if metric in average_uninferred:
					f.write('{}\t'.format(np.mean(average_uninferred[metric])))





if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# settings
	parser.add_argument('--gold_file', default='./gold/wire57_test.tsv')
	parser.add_argument('--prediction_file', default='./predictions/multi2oie_predictions/multi2oie_lsoie_wire57_predictions.tsv')
	parser.add_argument('--prediction_name', default=None)
	parser.add_argument('--sent_level_out_file', default=None)
	parser.add_argument('--information_dict_out_file', default=None)
	parser.add_argument('--predictedsent2goldsent', default='./gold/wire57_tokenized2nontokenized_sents.tsv')

	main_args = parser.parse_args()

	if main_args.prediction_name == None:
		main_args.prediction_name = main_args.prediction_file.split('/')[-1].replace('_predictions.tsv', '').replace('carb_', '')

	if main_args.sent_level_out_file == None:
		main_args.sent_level_out_file = '../../abstractive_openie/results/' + main_args.prediction_name + '_sent_stats.tsv'

	if main_args.information_dict_out_file == None:
		main_args.information_dict_out_file = '../../abstractive_openie/results/' + main_args.prediction_name + '_information_dict.tsv'

	predictedsent2goldsent = None
	if main_args.predictedsent2goldsent:
		predictedsent2goldsent = read_predictedsent2goldsent(main_args.predictedsent2goldsent)

	sent2gold = read_extraction_file(main_args.gold_file, 0, predictedsent2goldsent)

	information_dict = {}
	sent2predictions = read_extraction_file(main_args.prediction_file, 1, predictedsent2goldsent)
	st_p, st_r, st_f1 = calculate_sentence_tuple_score(sent2gold, sent2predictions, '{}_sent2tuple'.format(main_args.prediction_name), information_dict, sota_model, main_args.sent_level_out_file)
	ct_p, ct_r, ct_f1 = calculate_combined_tuple_tuple_score(sent2gold, sent2predictions, '{}_combined_tuple2tuple'.format(main_args.prediction_name), information_dict, sota_model, main_args.sent_level_out_file)
	tt_p, tt_r, tt_f1 = calculate_tuple_tuple_score(sent2gold, sent2predictions, '{}_tuple2tuple'.format(main_args.prediction_name), information_dict, sota_model, main_args.sent_level_out_file)
	with open('entailment_temp_file.tsv', 'a') as f:
		f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(main_args.prediction_name, st_p, st_r, st_f1, ct_p, ct_r, ct_f1, tt_p, tt_r, tt_f1))
	write_information_dict(information_dict, main_args.information_dict_out_file)
