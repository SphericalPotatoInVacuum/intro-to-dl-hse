all: process tfidf w2v

tfidf:
	python3 text.py train tfidf

w2v:
	python3 text.py train w2v

process:
	python3 text.py preprocess -i data/train.csv -o data/processed_train.pickle
	python3 text.py preprocess -i data/test.csv -o data/processed_test.pickle
