# Abstractive Open IE

This repository contains the code for the paper:\
Abstractive OpenIE\
Kevin Pei, Ishan Jindal, Kevin Chen-Chuan Chang\
EMNLP 2023

## Overview

Open Information Extraction (OpenIE) is the task of extracting relation tuples from plain text. In its simplest form, OpenIE extracts information in the form of tuples consisting of subject(S), predicate(P), object(O), and any additional arguments(A).
Abstractive OpenIE aims to expand on this task by including the extraction of inferred relations, which are relations where the tokens of the predicate or arguments do not exist within the sentence.

## Data
https://drive.google.com/file/d/1j1DyVu9-U2fn2E_MKGH8xql3OBLrMbk8/view?usp=sharing

Training data include: 
```oie4_train.tsv``` Base OIE4 training data
```de_backtranslated_oie4.tsv``` OIE4 with German back translated sentences
```sure_augmented_oie4.tsv``` OIE4 augmented with SuRE extractions
```de_backtranslated_sure_augmented_oie4.tsv``` OIE4 with both German backtranslated sentences and augmented by SuRE extractions

Dev sets include:
```lsoie_wiki_dev.tsv```

Test sets include: 
```WiRe57_test.txt```
```Re-OIE2016_test.txt```
```CaRB_test.txt```
```lsoie_wiki_test.tsv```
```CQ-W_QA_test.txt``` The test set used to evaluate on the downstream QUEST task

###Usage

```
python backtranslation.py [--FLAGS]
```

Creates paraphrases of sentences using back translation on the given training file. The language used is German.

```
python sure_data_creator.py [--FLAGS]
```

Creates inputs suitable for use with SuRE. The SuRE model can be accessed in the following github repository:
https://github.com/luka-group/SuRE

## Models
https://drive.google.com/file/d/1AZUmniWQ3irEXZkDbO5kbMZJztFSjylQ/view?usp=sharing

Models include: 
```abstractive_oie4``` T5 fine-tuned on base OIE4 training data
```abstractive_de_backtranslated_oie4``` T5 fine-tuned on OIE4 with German back translated sentences
```abstractive_sure_augmented_oie4``` T5 fine-tuned on OIE4 augmented with SuRE extractions
```abstractive_de_backtranslated_sure_augmented_oie4``` T5 fine-tuned on OIE4 with both German backtranslated sentences and augmented by SuRE extractions

### Usage

```
python main.py [--FLAGS]
```

Can be run in ```train``` or ```test``` mode
```--train_file``` and ```--dev_file``` are only necessary for training, and ```--test_file``` and ```--prediction_file``` are only necessary for testing


## Evaluation

We provide code for entailment-based metrics. They are sentence-tuple entailment, combined tuple-tuple entailment, and tuple-tuple entailment.

### Usage

```
python entailment_eval.py [--FLAGS]
```
