from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from scipy.special import softmax

advanced_bert_type = 'cross-encoder/nli-deberta-v3-base'
sota_model = CrossEncoder(advanced_bert_type)

tokenizers = {'en2de': AutoTokenizer.from_pretrained("facebook/wmt19-en-de"), 'de2en': AutoTokenizer.from_pretrained("facebook/wmt19-de-en"), 
			  'en2ru': AutoTokenizer.from_pretrained("facebook/wmt19-en-ru"), 'ru2en': AutoTokenizer.from_pretrained("facebook/wmt19-ru-en")}
translation_models = {'en2de': AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt19-en-de"), 'de2en': AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt19-de-en"),
					  'en2ru': AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt19-en-ru"), 'ru2en': AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt19-ru-en")}


def read_extraction_file(filename, is_prediction, predictedsent2goldsent=None):
	sent2extraction = {}
	with open(filename, 'r') as f:
		for line in f:
			splitLine = line.strip().split('\t')
			if predictedsent2goldsent:
				sentence = predictedsent2goldsent[splitLine[0]]
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
			extraction = {'arg1': arg1, 'pred': predicate, 'arg2': ' '.join(arg2s), 'relation': '{} {} {}'.format(arg1, predicate, ' '.join(arg2s))}
			if sentence not in sent2extraction:
				sent2extraction[sentence] = {'tuples': []}
			sent2extraction[sentence]['tuples'].append(extraction)

	for sentence in sent2extraction:
		sent2extraction[sentence]['combined_tuple'] = '. '.join([extraction['relation'] for extraction in sent2extraction[sentence]['tuples']])

	return sent2extraction

label2prob = {'contradiction': 0, 'neutral': 2, 'entail': 1}
label_mapping = ['contradiction', 'entailment', 'neutral']


def calculate_entailment(hypothesis, gold):
	examples = [(hypothesis_sent, gold_sent) for hypothesis_sent, gold_sent in zip(hypothesis, gold)]
	
	scores = sota_model.predict(examples)
	probs = softmax(scores, axis=1)[:, label2prob['entail']]
	labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
	return labels, probs


def check_entailment(paraphrases, sent2extraction, check_backward_entailment=False, is_strict=False):
	id2sent = {}
	id2tuple_id = {}
	i = 0
	forward_hypothesis = []
	forward_gold = []
	for sent, paraphrase in paraphrases:
		for tuple_id, extraction in enumerate(sent2extraction[sent]['tuples']):
			id2sent[i] = sent
			id2tuple_id[i] = tuple_id
			forward_hypothesis.append(paraphrase)
			forward_gold.append('{} {} {}.'.format(extraction['arg1'], extraction['pred'], extraction['arg2']))
			i += 1

	forward_entailment_labels, probs = calculate_entailment(forward_hypothesis, forward_gold)
	

	unentailing_paraphrases = set()
	if is_strict:
		for i, label in enumerate(forward_entailment_labels):
			if 'entail' not in label:
				unentailing_paraphrases.add(forward_hypothesis[i])

	if check_backward_entailment:
		back_hypothesis = []
		backward_gold = []
		for sent, paraphrase in paraphrases:
			back_hypothesis.append(sent2extraction[sent]['combined_tuple'])
			backward_gold.append(paraphrase)
		backward_entailment_labels, probs = calculate_entailment(back_hypothesis, backward_gold)
		for i, label in enumerate(backward_entailment_labels):
			if 'entail' not in label:
				unentailing_paraphrases.add(backward_gold[i])


	lines = []
	for i, label in enumerate(forward_entailment_labels):
		if 'entail' in label and forward_hypothesis[i] not in unentailing_paraphrases:
			extraction = sent2extraction[id2sent[i]]['tuples'][id2tuple_id[i]]
			lines.append('{}\t{}\t{}\t{}\n'.format(forward_hypothesis[i], extraction['pred'], extraction['arg1'], extraction['arg2']))

	return lines


def read_sentence_file(filename):
	sents = []
	with open(filename, 'r') as f:
		for line in f:
			sents.append(line.strip())
	return sents

def generate_paraphrase(lang, sentence):
	input_ids = tokenizers['en2{}'.format(lang)](sentence, return_tensors="pt").input_ids
	foreign_ids = translation_models['en2{}'.format(lang)].generate(input_ids)[0]
	foreign_output = tokenizers['en2{}'.format(lang)].decode(foreign_ids, skip_special_tokens=True)

	foreign_ids = tokenizers['{}2en'.format(lang)](foreign_output, return_tensors="pt").input_ids
	output_ids = translation_models['{}2en'.format(lang)].generate(foreign_ids)[0]
	en_output = tokenizers['{}2en'.format(lang)].decode(output_ids, skip_special_tokens=True)

	return (sentence, en_output)


def generate_paraphrases(lang, sentences):
	input_ids = tokenizers['en2{}'.format(lang)](sentences, return_tensors="pt", padding=True).input_ids
	foreign_ids = translation_models['en2{}'.format(lang)].generate(input_ids)#[0]
	foreign_outputs = []
	for foreign_id in foreign_ids:
		foreign_outputs.append(tokenizers['en2{}'.format(lang)].decode(foreign_id, skip_special_tokens=True))

	foreign_ids = tokenizers['{}2en'.format(lang)](foreign_outputs, return_tensors="pt", padding=True).input_ids
	output_ids = translation_models['{}2en'.format(lang)].generate(foreign_ids)#[0]
	en_outputs = []
	for output_id in output_ids:
		en_outputs.append(tokenizers['en2{}'.format(lang)].decode(output_id, skip_special_tokens=True))

	return [(sentence, paraphrase) for sentence, paraphrase in zip(sentences, en_outputs)]


def paraphrase_sent_file(lang, sent_file):
	sents = read_sentence_file(sent_file)
	paraphrases = []
	for sent in tqdm(sents):
		paraphrases.append(generate_paraphrase(lang, sent))
	return paraphrases


def paraphrase_extraction_file(lang, extraction_file, out_files, entailment_parameters):
	for out_file in out_files:
		with open(out_file, 'w') as f:
			f.write('')
	sent2extraction = read_extraction_file(extraction_file, is_prediction=0)
	sents = [sent for sent in sent2extraction.keys()]
	
	paraphrases = []
	i = 0
	for sent in tqdm(sents):
		paraphrases.append(generate_paraphrase(lang, sent))

		i += 1
		if i % 100 == 0:
			for j, entailment_parameter in enumerate(entailment_parameters):
				lines = check_entailment(paraphrases[i-100:i], sent2extraction, entailment_parameter['check_backward_entailment'], entailment_parameter['is_strict'])

				with open(out_files[j], 'a') as f:
					for line in lines:
						f.write(line)
						
		elif i == len(sents):
			for j, entailment_parameter in enumerate(entailment_parameters):
				lines = check_entailment(paraphrases[i-(i%100):i], sent2extraction, entailment_parameter['check_backward_entailment'], entailment_parameter['is_strict'])

				with open(out_files[j], 'a') as f:
					for line in lines:
						f.write(line)

	return paraphrases



if __name__ == '__main__':
	for lang in ['de']:
		out_files = []
		entailment_parameters = []
		out_files.append('./results/{}_back_translated_oie4.tsv'.format(lang))
		entailment_parameters.append({'check_backward_entailment': True, 'is_strict': True})

		paraphrases = paraphrase_extraction_file(lang, './oie4_train.tsv', out_files, entailment_parameters)
