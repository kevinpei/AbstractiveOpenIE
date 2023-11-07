import spacy
import json

nlp = spacy.load("en_core_web_sm")
non_noun_tags = [u'PRON']

def read_train(filename):
	sents = []
	with open(filename, 'r') as f:
		for line in f:
			sent = line.split('\t')[0]
			pred = line.split('\t')[1]
			arg1 = line.split('\t')[2]
			arg2 = ' '.join(line.strip().split('\t')[3:])
			if len(sents):
				if sent != sents[-1]['sent']:
					sents.append({'sent': sent, 'extractions': []})
			else:
				sents.append({'sent': sent, 'extractions': []})
			sents[-1]['extractions'].append({'pred': pred, 'arg1': arg1, 'arg2': arg2})
	return sents


def create_noun_phrase_pairs_from_sent(sent):
	subj_obj_pairs = []
	doc = nlp(sent)
	noun_phrases = []
	for chunk in doc.noun_chunks:
		if chunk.root.pos_ not in non_noun_tags:
			noun_phrases.append(chunk)
	for i in range(len(noun_phrases)):
		if i < len(noun_phrases) - 1:
			for j in range(i+1, min(i+3, len(noun_phrases))):
				subj_obj_pairs.append((noun_phrases[i], noun_phrases[j]))
		if i > 0:
			subj_obj_pairs.append((noun_phrases[i], noun_phrases[i-1]))

	return subj_obj_pairs


def create_training_data_dict_from_sents(filename, sents):
	data_points = []
	data_id = 0
	for sent in sents:
		subj_obj_pairs = create_noun_phrase_pairs_from_sent(sent['sent'])
		for pair in subj_obj_pairs:
			already_exists = False
			for extraction in sent['extractions']:
				if (pair[0].root.text in extraction['arg1'] and pair[1].root.text in extraction['arg2']) or (pair[0].root.text in extraction['arg2'] and pair[1].root.text in extraction['arg1']):
					already_exists = True
					break
			if already_exists:
				continue
			subj = pair[0].text
			obj = pair[1].text
			data_point = {}
			data_point['id'] = '{}_{}'.format(filename, data_id)
			data_point['subj'] = subj
			data_point['obj'] = obj
			data_point['relation'] = 'no_relation'
			data_point['target'] = '{} has no known relations to {}'.format(subj, obj)
			data_point['text'] = 'The head entity is {}. The tail entity is {}. {}'.format(subj, obj, sent['sent'])
			data_point['sent'] = sent['sent']
			data_points.append(data_point)
			data_id += 1

	return data_points


def create_training_data_from_file(filename, out_file):
	sents = read_train(filename)
	data_points = create_training_data_dict_from_sents(filename, sents)
	with open(out_file, 'w') as f:
		for data_point in data_points:
			data_point_str = json.dumps(data_point)
			f.write(data_point_str + '\n')


create_training_data_from_file('../../../datasets/lsoie_data/no_duplicate_lsoie_wiki_train.tsv', './base_oie_datasets/lsoie_wiki/lsoie_wiki_sure_data.json')
