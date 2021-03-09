# Question Classifier
question classifer

## pipeline
```
tokenization->word embedding->sentence vector->training the classifier
```

## commit msg
`[your task]: what you did in this commit`

e.g.: 'wordEmbedding: word2vec model initialize'

...

## run
```
cd src
```
dev training mode:
Leaving 10% of training set out as validation set
```
python3 question_classifier.py --dev --config [config-file-path]
```
training mode:
Train the model with the whole dataset
```
python3 question_classifier.py --train --config [config-file-path]
```
test mode:
Read an existing model and test it on TREC 10 dataset
```
python3 question_classifier.py --test --config [config-file-path]
```
search mode:
Searching hyper-params
```
python3 question_classifier.py --search --config [config-file-path]
```