all: process tfidf w2v

tfidf: process
	python3 text.py train tfidf

w2v: process
	python3 text.py train w2v

ft: process
	python3 text.py train ft

bert:
	python3 text.py train bert

process:
	python3 text.py preprocess data/train.csv data/processed_train.pickle
	python3 text.py preprocess data/test.csv data/processed_test.pickle
