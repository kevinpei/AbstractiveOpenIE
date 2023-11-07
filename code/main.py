import os

import transformers
import numpy as np
import nltk
import torch
import copy
from tqdm import tqdm
import argparse
from utils import preprocess_data, create_abstractive_file, create_dataset, compute_tuple_tuple_score, tokenizer, model_checkpoints

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_input = 256
max_target = 32
optim = 'adam'
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_checkpoints)
model.resize_token_embeddings(len(tokenizer))
collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)


def finetune_model(train_filename, val_filename, main_args):

	train = create_dataset(train_filename)
	val = create_dataset(val_filename)

	args = transformers.Seq2SeqTrainingArguments(
		output_dir='./models/{}'.format(main_args.model_name),
		evaluation_strategy='epoch',
		learning_rate=main_args.lr,
		per_device_train_batch_size=main_args.batch_size,
		per_device_eval_batch_size= main_args.batch_size,
		gradient_accumulation_steps=2,
		weight_decay=0.01,
		save_total_limit=2,
		save_strategy='no',
		num_train_epochs=3,
		predict_with_generate=True,
		eval_accumulation_steps=1,
		optim=optim
	)

	trainer = transformers.Seq2SeqTrainer(
		model, 
		args,
		train_dataset=train,
		eval_dataset=val,
		data_collator=collator,
		tokenizer=tokenizer,
		compute_metrics=compute_tuple_tuple_score
	)
	trainer.train()
	trainer.save_model('./models/{}'.format(main_args.model_name))


def read_test(filename):
	lines = []
	with open(filename, 'r') as f:
		for line in f.readlines():
			lines.append(line.strip())
	return lines


def write_predictions(sentences, predictions, out_path):
	with open(out_path, 'w') as f:
		for i, prediction in enumerate(predictions):
			f.write('{}\t{}\n'.format(sentences[i], prediction))


def generate_predictions(main_args):
	lines = read_test(main_args.test_file)
	model = transformers.AutoModelForSeq2SeqLM.from_pretrained('./models/{}'.format(main_args.model_name), 
                                                        local_files_only=True).to(device)

	args = transformers.Seq2SeqTrainingArguments(
		output_dir='./models/{}'.format(main_args.model_name),
		evaluation_strategy='epoch',
		learning_rate=main_args.lr,
		per_device_train_batch_size=main_args.batch_size,
		per_device_eval_batch_size= main_args.batch_size,
		gradient_accumulation_steps=2,
		weight_decay=0.01,
		save_total_limit=2,
		save_strategy='no',
		num_train_epochs=3,
		predict_with_generate=True,
		eval_accumulation_steps=1
	)

	trainer = transformers.Seq2SeqTrainer(
		model, 
		args,
		data_collator=collator,
		tokenizer=tokenizer,
		compute_metrics=compute_tuple_tuple_score
	)

	predictions = []
	sents = []
	for line in tqdm(lines):
		sentence = copy.deepcopy(line)
		model_inputs = tokenizer(main_args.predicate_prompt.replace('<sent>', sentence),  max_length=max_input, return_tensors="pt", padding='max_length', truncation=True)
		raw_preds = model.generate(input_ids=model_inputs["input_ids"].to(device),
									attention_mask=model_inputs["attention_mask"].to(device), num_beams=main_args.num_beams)
		predicates = [pred.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip() for pred in tokenizer.decode(raw_preds[0]).split('[pred]')]
		prior_relations = []
		for predicate in predicates:
			if len(predicate):
				model_inputs = tokenizer(main_args.arg_prompt.replace('<sent>', sentence).replace('<pred>', predicate),  max_length=max_input, return_tensors="pt", padding='max_length', truncation=True)
				raw_args = model.generate(input_ids=model_inputs["input_ids"].to(device),
											attention_mask=model_inputs["attention_mask"].to(device), num_beams=main_args.num_beams)
				args = [arg.strip() for arg in tokenizer.decode(raw_args[0]).replace('<s>', '').replace('</s>', '').replace('<pad>', '').replace('[arg1]', '').split('[arg2]')]

				arg1 = args[0]
				if len(args) > 1:
					arg2 = args[1]
				else:
					arg2 = ''

				prior_relations.append(arg1 + ' ' + predicate + ' ' + arg2)
				predictions.append('[arg1] {} [pred] {} [arg2] {} [et]'.format(arg1, predicate, arg2))
				sents.append(line)
			sentence += ' [pred] ' + predicate
	write_predictions(sents, predictions, main_args.predictions_file)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# settings
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--num_beams', type=int, default=5)
	parser.add_argument('--lr', type=float, default=1e-5)
	parser.add_argument('--batch_size', type=str, default=4)
	parser.add_argument('--mode', type=str, default='train')
	parser.add_argument('--cuda_device', type=str, default='0')
	parser.add_argument('--predicate_prompt', type=str, default='Extract the predicates from this sentence: <sent>')
	parser.add_argument('--arg_prompt', type=str, default='Extract the relation arguments for this predicate and this text: <pred> [SEP] <sent>')
	parser.add_argument('--model_name', type=str, default='t5_oie4')

	parser.add_argument('--train_file', type=str, default='./data/train/oie4_train.tsv')
	parser.add_argument('--dev_file', type=str, default='./data/dev/lsoie_wiki_dev.tsv')
	parser.add_argument('--test_file', type=str, default='./data/test/WiRe57_test.txt')
	parser.add_argument('--predictions_file', type=str, default='./predictions/t5_oie4_wire57_predictions.tsv')
	main_args = parser.parse_args()

	if main_args.mode == 'train':
		abstractive_train_file = main_args.train_file.replace('/train/', '/train/abstractive_')
		create_abstractive_file(main_args.train_file, abstractive_train_file, main_args)

		abstractive_dev_file = main_args.dev_file.replace('/dev/', '/dev/abstractive_')
		create_abstractive_file(main_args.dev_file, abstractive_dev_file, main_args)

		finetune_model(abstractive_train_file, abstractive_dev_file, main_args)

	elif main_args.mode == 'test':
		generate_predictions(main_args)